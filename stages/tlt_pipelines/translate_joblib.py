import os

import torch
from torch.utils.data import DataLoader

from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context

from transformers import AutoModelForSeq2SeqLM

from datasets import (
    Dataset as HFDataset, 
    load_from_disk,
    load_dataset,
    concatenate_datasets
)

from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk

import glob
from functools import partial

import tqdm
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Perform distributed inference")

    parser.add_argument(
        "--root_dir",
        type=str,
    )

    parser.add_argument(
        "--data_files",
        type=str,
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
    )

    parser.add_argument(
        "--base_save_dir",
        type=str,
    )

    parser.add_argument(
        "--joblib_temp_folder",
        type=str,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
    )

    parser.add_argument(
        "--total_procs",
        type=int
    )

    parser.add_argument(
        "--devices",
        type=lambda x: [ int(idx.strip()) for idx in x.split(",") if idx and len(idx.strip()) ],
        required=True,
    )

    args = parser.parse_args()

    return args

def find_optimal_batch_size(dataset_length, min_batch_size, max_batch_size):
    """
    Finds the optimal batch size within a given range that minimizes truncation of a dataset.
    
    Parameters:
    dataset_length (int): The size of the dataset.
    min_batch_size (int): The minimum allowed batch size.
    max_batch_size (int): The maximum allowed batch size.
    
    Returns:
    int: The optimal batch size within the specified range that minimizes dataset truncation.
    """
    # Ensure the search range is valid
    if min_batch_size > max_batch_size or dataset_length <= 0:
        raise ValueError("Invalid parameters: Ensure dataset_length > 0 and min_batch_size <= max_batch_size.")
    
    optimal_batch_size = min_batch_size
    min_remainder = dataset_length % min_batch_size

    for batch_size in range(min_batch_size + 1, max_batch_size + 1):
        current_remainder = dataset_length % batch_size
        
        # Optimal batch size found if no truncation
        if current_remainder == 0:
            return batch_size
        
        # Update optimal batch size if a smaller remainder is found
        if current_remainder < min_remainder:
            min_remainder = current_remainder
            optimal_batch_size = batch_size
    
    return optimal_batch_size

def padding_collator(
    batch, 
    keys_to_pad=[
            ("input_ids", 1), 
            ("attention_mask", 0),
        ]
    ):

    batch_out = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
        for key in batch_out.keys():
            batch_out[key] += [sample[key]]
    
    for key, value_to_pad_with in keys_to_pad:

        len_list = list(map(lambda x: len(x), batch_out[key]))

        padding_length = max(len_list)
        tensor_list = []
        for i, x in enumerate(batch_out[key]):

            if len(x) < padding_length:
                tensor_list += [torch.tensor([value_to_pad_with]*(padding_length - len_list[i]) + x)]
            else:
                tensor_list += [torch.tensor(x)]

        batch_out[key] = torch.stack(tensor_list)

    return batch_out



def save_checkpoint(index, ds, filename="checkpoint.pth.tar"):
    temp_translated_dataset = 'temp_translated_dataset'
    ds_dir = f'{temp_translated_dataset}_{index}'
    ds.save_to_disk(ds_dir)
    if os.path.isfile(filename):
        remove_checkpoint(filename)
        print('checkpoint metadata saved')
        torch.save({'batch_idx': index, 'ds_dir': ds_dir}, filename)
        
    else:
        print('checkpoint saving first time in this directory')
        torch.save({'batch_idx': index, 'ds_dir': ds_dir}, filename)

def remove_checkpoint(filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        ds_dir = torch.load(filename)['ds_dir']
        print(f'Removing dataset checkpoint {ds_dir}...')
        if os.path.isdir(ds_dir):
            os.system(f'rm -rf {ds_dir}/*')
            os.rmdir(ds_dir)
            os.remove(filename)

def load_checkpoint(filename="checkpoint.pth.tar"):
    print('Loading Checkpoint ', filename)
    if os.path.isfile(filename):
        data = torch.load(filename)
        print(data)
        ds_dir = data['ds_dir']
        ds = load_from_disk(ds_dir)
        print('Dataset loaded from checkpoint')
        print(ds)
        return {'batch_idx': data['batch_idx'], 'run_ds': ds}
    else:

        return None

def resume_from_checkpoint(data_loader, checkpoint):
    if checkpoint is not None:
        print('Resuming from dataset...')
        start_batch = checkpoint['batch_idx'] + 1
        run_ds = checkpoint['run_ds']
    else:
        print('Starting from scratch...')
        start_batch = 0
        run_ds = HFDataset.from_dict({ key: [] for key in ["doc_id", "sids", "sub_strs", "tlt_idx", "placeholder_entity_map", "translation_ids"] })

    return start_batch, run_ds

def _mp_fn(ds, base_save_dir, batch_size, rank, device, world_size, procs_to_write):
    device = f"cuda:{device}"

    rank_ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

    data_loader = torch.utils.data.DataLoader(
        rank_ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=8,
        collate_fn=padding_collator,
        multiprocessing_context=get_context('loky')
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True).eval().to(device)

    checkpoint = load_checkpoint(f"checkpoint_rank_{rank}.pth.tar")
    start_batch, run_ds = resume_from_checkpoint(data_loader, checkpoint)

    print(f'Start batch: {start_batch}\nlen of data loader: {len(data_loader)}\nrun_ds: ')
    print(run_ds)

    with torch.no_grad():
        for idx, batch in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader), unit=f"ba: {batch_size} samples/ba"):
            if start_batch >= idx:
                continue
            input_ids = batch["input_ids"].to(device)
            outputs = model.generate(input_ids=input_ids, num_beams=1, num_return_sequences=1, max_length=256, do_sample=False)

            run_ds = concatenate_datasets([
                run_ds,
                HFDataset.from_dict({
                    "doc_id": batch["input_ids"], 
                    "sid": batch["sids"], 
                    "sub_str": batch["sub_strs"], 
                    "tlt_idx": batch["tlt_idx"], 
                    "placeholder_entity_map": batch["placeholder_entity_map"],
                    "translated_input_ids": outputs.to("cpu"),
                    "tlt_file_loc": batch["tlt_file_loc"],
                })
            ])

            # Save checkpoint after every batch (or less frequently if desired)
            if idx != start_batch and idx % 3000 ==0:
                print(f"Saving checkpoint at batch {idx}")
                save_checkpoint(idx, run_ds, filename=f"checkpoint_rank_{rank}.pth.tar")
                

    save_dir = os.path.join(base_save_dir, f"rank_{rank}-device_{device.split(':')[0]}")
    os.makedirs(save_dir, exist_ok=True)
    run_ds.save_to_disk(save_dir, num_proc=procs_to_write)
    print('Removing temporary translated dataset checkpoint...')
    remove_checkpoint(f"checkpoint_rank_{rank}.pth.tar")


    return True


if __name__ == "__main__":

    args = parse_args()

    ds = load_dataset(
        "arrow",
        data_files=glob.glob(args.data_files),
        num_proc=args.total_procs,
        cache_dir=args.cache_dir,
        split="train"
    )

    args.batch_size = find_optimal_batch_size(len(ds), args.batch_size, 390)

    batch_status = Parallel(
        n_jobs=len(args.devices),
        verbose=0, 
        backend="loky",
        batch_size="auto",
        pre_dispatch='n_jobs',
        temp_folder=args.joblib_temp_folder,
    )(
        delayed(_mp_fn)(
            ds=ds,
            base_save_dir=args.base_save_dir,
            batch_size=args.batch_size,
            rank=idx,
            device=device,
            world_size=len(args.devices),
            procs_to_write=args.total_procs//len(args.devices)
        )  for idx, device in enumerate(args.devices)
    )

    
