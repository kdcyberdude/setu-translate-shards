from flask import Flask, request, jsonify
from healthcheck import HealthCheck
from multiprocessing import Process, Queue, set_start_method
import time
import json
from stages.document import Document
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from datasets import Dataset, concatenate_datasets, load_dataset, disable_progress_bar
from huggingface_hub import HfApi, HfFolder
import torch
from tqdm import tqdm
from functools import partial
from datatrove.pipeline.readers import ParquetReader
import re
import pandas as pd
import os
import psutil
import logging

print('HF_TOKEN ...', HfFolder.get_token())
test_text = '''– Computer viruses are parasitic programs which are able to replicate themselves, attach themselves to other executables in the computer, and perform some unwanted and often malicious actions. A virus is not able to spread itself to another computers, some user actions are needed for it to infect a new computer. Downloading and running software from untrusted sources, inserting an USB drive without a previous scan–remember always disable the AutoRun feature for the drives as CD-ROMs, DVD-ROMs– , downloading and running emails or IM attachments even from known persons, can put you in the nasty situation to have an infected computer. Always when you deal with these situations and to prevent computer infections, scan before to run.
The best scanners in my opinion are multi-engine online scanners like virustotal.com or novirusthanks.org. The links of these scanners and many more are on the home page.
The main three features of a virus are :
– the replication mechanism search and find other executable files in your computer, check if the files are already infected–it has a special mechanism for that and if the file is clean then append itself to the file. It can append to the front, middle or end of the executable file thus changing the file size. This is also the reason why the number of new created viruses decreased in the last years, the AntiViruses has a very simple mechanism for “checking and compare” the files size –checksums at different period of times and a file bigger in size than at a previous date is a sign of infection.
A special category of viruses are “Bacteria” viruses, they replicate themselves so quickly and in a such percentage that the harddisk will run very soon out of free space.
– a trigger is designed to activate a task of the virus, as displaying strange messages, deleting files, sending emails, begin the replicate process or whatever the programmer write in his malicious code. The trigger can be a certain date in calendar–formerly know as Time Bombs, the time when some event occur, opening a certain program or other users actions. The trigger is very important for the virus spreading, because once infected the user will notice nothing strange in his computer, and will continue to spread the virus without to suspect anything. Other reason of this delaying of infection symptoms is for viruses to hide its tracks, the user simply does not know when and how it get infected.
– the task or “payload” can differ from inoffensive ones like displaying joke messages, to deleting or editing important system files like hosts file , deleting or editing registry entries, sometimes making the computer unbootable.
Using polymorphic engines, the viruses change the “virus signature”–their binary code each time when they infect a new computer making very difficult for AntiViruses to detect them using traditional “signature based” scanners.
Macro Viruses can attach themselves to the executable portion of a spreadsheet documents in AutoOpen, AutoClose, AutoExit, or other file macros. The words processors are the most affected by these viruses, so to prevent the computer infections, always perform an AntiVirus scan for documents received as emails attachments, or received by another methods from another computers.
– Computer worms are a special category of viruses with a very important feature added : they can spread themselves between computers, without the user interaction, exploiting some networks vulnerabilities or facilities as network shares or remote file executions. It’s recommended by some experts to disable the Windows Remote Assistance feature, seeing this as a possibly vulnerability. Once it infect a computer it looks forward for other computers connected to the network–LAN or Internet continuing to search for possibly victims.
– Trojans are malicious executable files, masquerading as beneficial programs and tricking users to run their code. Very often they are embedded into other programs as setup or installers file and shared into the forums or blogs as pirated software(warez), so when the user run the installer of a program, the trojan will run in parallel infecting the computer. It’s a server-client software and once infecting a computer, it gives to the hacker where it connects the full power over the computer.
The hacker can see screenshots of the victims computer, can see the webcam in real time, can download and upload files or run other malware.
They are very trendy in nowdays and deserve a special attention, so a more detailed description of this type of malware will be given in the part 2 of this article.
– Spyware is a malicious code able to gather private data from an infected computer and send it to the hacker. The data can be passwords, credit card numbers, login credentials and other private data. They accomplish their mission by using various mechanisms for decrypting passwords previously saved in web-browsers, keyloggers or screenshots. The computer user get infected by spyware in several ways by downloading and running “fake antiviruses” or “registry cleaners” or visiting malicious sites t'''

app = Flask(__name__)
health = HealthCheck(app=app, path="/hc")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
disable_progress_bar()

# Create queues for each stage with limited size
preprocess_queue = Queue(maxsize=128)
translate_queue = Queue(maxsize=128) 
postprocess_queue = Queue(maxsize=256)
save_queue = Queue(maxsize=256)

ip = IndicProcessor(inference=True)
tokenizer = IndicTransTokenizer(direction="en-indic")
placeholder_entity_maps = list(map(lambda ple_map: json.dumps(
    ple_map), ip.get_placeholder_entity_maps(clear_ple_maps=True)))

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
                tensor_list += [torch.tensor([value_to_pad_with]
                                             * (padding_length - len_list[i]) + x)]
            else:
                tensor_list += [torch.tensor(x)]

        batch_out[key] = torch.stack(tensor_list)

    return batch_out

def decode(batch, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    p_batch = dict()
    input_ids = batch.pop("translated_input_ids")
    outputs = tokenizer.batch_decode(input_ids, src=False)
    p_batch["translated"] = ip.postprocess_batch(
        outputs, lang=tgt_lang, placeholder_entity_maps=[{}] * len(outputs))
    return p_batch | {
        "translated_input_ids": input_ids,
    }


def preprocess(text):
    text = text.strip()
    if text == '':
        return None
    doc = Document(text=text,)
    doc_dict = doc.get_templated_document_attrs()
    doc_id = doc_dict["doc_id"]
    result = {}
    sids = json.loads(doc_dict["sids"])
    substrs = json.loads(doc_dict["sub_strs"])
    result['doc_id'] = [doc_id] * len(sids)
    result['sid'] = sids
    result['substr'] = substrs

    padding = 'max_length'
    src_lang = 'eng_Latn'
    tgt_lang = 'pan_Guru'
    batch_size = 1024

    input_ids_all, attention_mask_all = [], []
    for i in range(0, len(result['substr']), batch_size):
        batch = result['substr'][i:i+batch_size]
        sentences = ip.preprocess_batch(
            batch,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        input_ids, attention_mask = tokenizer(
            sentences, src=True, padding=padding, truncation=True if padding == 'max_length' else False, return_tensors='pt').values()
        input_ids_all.extend(input_ids)
        attention_mask_all.extend(attention_mask)

    result['input_ids'] = input_ids_all
    result['attention_mask'] = attention_mask_all
    return result


def translate(data, model):
    ds = Dataset.from_dict(data)
    batch_size = 64  # expectedly will use 10 - 5.5 GB of GPU memory
    device = 'cuda:0'

    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, drop_last=False, num_workers=1, collate_fn=padding_collator)

    run_ds = Dataset.from_dict(
        {key: [] for key in ["doc_id", "sid" "sub_str",
                             "translated_input_ids", "placeholder_entity_map"]},
    )
    with torch.no_grad():
        # for idx, batch in tqdm(enumerate(data_loader, 0), total=len(data_loader), unit=f"ba: {batch_size} samples/ba"):
        for idx, batch in enumerate(data_loader, 0):
            input_ids = batch["input_ids"].to(device)
            outputs = model.generate(input_ids=input_ids, num_beams=1,
                                     num_return_sequences=1, max_length=256, do_sample=False)

            run_ds = concatenate_datasets([
                run_ds,
                Dataset.from_dict({
                    "doc_id": batch["doc_id"],
                    "sid": batch["sid"],
                    "sub_str": batch["substr"],
                    "translated_input_ids": outputs.to("cpu"),
                })
            ])

    return run_ds


def postprocess(ds):
    src_lang = 'eng_Latn'
    tgt_lang = 'pan_Guru'

    decoded_ds = ds.map(
        partial(
            decode,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        ),
        batched=True,
    )
    if len(decoded_ds["doc_id"]) == 0:
        return None
    decoded_ds = {
        "id": decoded_ds["doc_id"][0],
        "eng": decoded_ds["sub_str"],
        "pan": decoded_ds["translated"],
    }
    return decoded_ds

def worker(task_queue, next_queue, task_func):
    while True:
        data = task_queue.get()  # This will block until an item is available
        result = task_func(data)
        if next_queue is not None:
            next_queue.put((result))
        else:
            save_queue.put((result))

def translate_worker(translate_queue, postprocess_queue, translate_func):
    device = 'cuda:0'
    model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True).eval().to(device)
    print('Model loaded ...')
    with open('model_loaded.txt', 'w') as f:
        f.write('Model loaded ...')
    while True:
        data = translate_queue.get()  
        result = translate_func(data, model)
        postprocess_queue.put((result))


def save_worker(save_queue, output_file_path):
    batch_size = 500  # Define a batch size for writing to disk
    buffer = []
    start_time = time.time()
    count = 0

    while True:
        result = save_queue.get()
        if result == 'NONENONENONE':
            if buffer:
                with open(output_file_path, 'a') as output_file:
                    output_file.write('\n'.join(buffer) + '\n')

        buffer.append(json.dumps(result))
        count += 1

        if count % batch_size == 0:
            with open(output_file_path, 'a') as output_file:
                output_file.write('\n'.join(buffer) + '\n')
            buffer = []  # Clear the buffer after writing

        if count % 100 == 0:
            print(f"Processed {count} documents in {(time.time() - start_time)/60:.2f} minutes")



def monitor_queues(queues):
    """Monitor queues and proceed when all are empty."""
    while True:
        all_empty = all(queue.empty() for queue in queues)
        if all_empty:
            print("All queues are empty. Proceeding...")
            break
        else:
            print("Waiting for all queues to empty...")
            time.sleep(15)  # Polling interval

def extract_shard_number(glob_pattern):
    match = re.search(r'train-(\d+)-of-\d+\.parquet', glob_pattern)
    if match:
        return match.group(1)
    else:
        raise ValueError("glob_pattern does not match the expected format")
    
def convert_jsonl_to_parquet(jsonl_file_path, parquet_file_path):
    df = pd.read_json(jsonl_file_path, lines=True)
    df.to_parquet(parquet_file_path, index=False)

output_file_path = 'processed_parquet_file.jsonl'

# Start worker processes
Process(target=worker, args=(preprocess_queue, translate_queue, preprocess), daemon=False).start()
Process(target=translate_worker, args=(translate_queue, postprocess_queue, translate), daemon=False).start()
Process(target=worker, args=(postprocess_queue, None, postprocess), daemon=False).start()
Process(target=save_worker, args=(save_queue, output_file_path), daemon=False).start()

def model_loaded():
    is_ready = os.path.exists('model_loaded.txt')
    return is_ready, "model_loaded"

health.add_check(model_loaded)

@app.route('/hc', methods=['GET'])
def readiness_probe():
    print('Readiness probe hit ...')
    return health.run()

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    start = time.time()
    data = request.json['data']
    print('TRANSLATE ENDPOINT HIT ...', data)
    data_folder = data['data_folder']
    glob_pattern = data['glob_pattern']

    if os.path.exists(output_file_path):
        print('Removing existing jsonl file ...')
        os.remove(output_file_path)
    
    shard_number = extract_shard_number(glob_pattern)

    data_reader = ParquetReader(data_folder, glob_pattern=glob_pattern)
    count = 0
    
    for doc in tqdm(data_reader()):
        preprocess_queue.put((doc.text))
        # if count > 100:
        #     break
        count += 1


    token = os.getenv('HF_TOKEN')
    print('HF_TOKEN ...', token)
    print('HF_TOKEN ...', HfFolder.get_token())
    time.sleep(10)
    monitor_queues([preprocess_queue, translate_queue, postprocess_queue, save_queue])
    save_queue.put('NONENONENONE')

    # Convert JSONL to Parquet
    parquet_file_path = f'processed_shard_{shard_number}.parquet'
    convert_jsonl_to_parquet(output_file_path, parquet_file_path)
    print('Uploading parquet file to hub...')
    api = HfApi()
    commit_info = api.upload_file(
        path_or_fileobj=parquet_file_path,
        path_in_repo=os.path.join('data', os.path.basename(parquet_file_path)),
        repo_id='kdcyberdude/fineweb-edu-pa',
        repo_type='dataset',
        token=HfFolder.get_token()
    )
    end = time.time()
    print(f'Total time taken: {(end - start) / 60}M')

    print(f'File uploaded to hub with commit info: {commit_info}')
    return jsonify({"result": f'{parquet_file_path} file uploaded to hub with commit info: {commit_info}'})

if __name__ == '__main__':
    set_start_method('spawn', force=True)

    app.run(host="::", port=5000, debug=False)
