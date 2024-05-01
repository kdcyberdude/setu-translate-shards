#!/bin/zsh

# Function to hold the terminal open on error
error_handler() {
    echo "Error occurred in script execution. Holding the terminal open. Check the logs above for more details."
    read -p "Press enter to continue..."
}

# Trap any error with the error_handler function
trap error_handler ERR

# Stop execution on any error
set -e

# List directories following the pattern "translated_*"
dirs=$(ls -d ../../setu_output/stories/translated_*)

# Extract the indices, sort them numerically, and get the last one
last_index=$(echo "$dirs" | sed 's/.*translated_//' | sort -n | tail -1)

# Add 1 to the last index
next_index=$((last_index + 1))
echo "next_index :"
echo $next_index

start_shard=$next_index
end_shard=2

# Define variables
base_save_dir=/mnt/sea/setu/
cache_dir=/mnt/sea/setu_cache/
source_type=wikihow
text_col=text
repo_base_path=/home/kd/Desktop/proj/dec/setu-translate

# Set RM_STAR_SILENT to avoid confirmation prompt for 'rm *'
setopt RM_STAR_SILENT

source /home/kd/anaconda3/bin/activate translate-env

echo "----------------------------------------------" >> process_log.txt

for shard_number in {$start_shard..$end_shard}; do

    parquet_file=/mnt/sea/dolma/parquet_shards/wikihow/${shard_number}.parquet

    # Log start time
    echo "Starting process at $(date) ${source_type} | ${parquet_file}" >> process_log.txt
    echo "Starting process at $(date) ${source_type} | ${parquet_file}"

    file_name_without_extension=$(basename "$parquet_file" .parquet)
    echo "Processing shard index: $file_name_without_extension"
    echo "Processing shard index: $file_name_without_extension" >> process_log.txt

    # Templating
    echo "Starting templating..." >> process_log.txt
    python ${repo_base_path}/stages/perform_templating.py --glob_path $parquet_file --base_save_path "${base_save_dir}/templating/doc_csvs" --save_path "${base_save_dir}/templated" --text_col ${text_col} --source_type ${source_type} --translation_type sentence --use_cache False --split "train" --batch_size 2084 --total_procs 24 --cache_dir_for_original_data "${cache_dir}/templating/cache"
    echo "Templating completed successfully." >> process_log.txt

    # Create Global Sentence Level Dataset
    echo "Starting global sentence level dataset creation..." >> process_log.txt
    python ${repo_base_path}/stages/create_global_ds.py --paths_data "${base_save_dir}/templated/*.arrow" --global_sent_ds_path "${base_save_dir}/sentences" --batch_size 2084 --total_procs 24 --cache_dir "${cache_dir}/global_sent_ds/cache"
    echo "Global sentence level dataset creation completed successfully." >> process_log.txt

    # Binarize
    echo "Starting binarization..." >> process_log.txt
    python ${repo_base_path}/stages/binarize.py --data_files "${base_save_dir}/sentences/*.arrow" --binarized_dir "${base_save_dir}/binarized_sentences" --batch_size 2084 --total_procs 24 --padding max_length --src_lang eng_Latn --tgt_lang pan_Guru --return_format pt --cache_dir "${cache_dir}/binarize/cache"
    echo "Binarization completed successfully." >> process_log.txt

    # Translate
    echo "Starting translation..." >> process_log.txt
    python ${repo_base_path}/stages/tlt_pipelines/translate_joblib.py --data_files "${base_save_dir}/binarized_sentences/*.arrow" --base_save_dir "${base_save_dir}/model_out" --joblib_temp_folder "/mnt/sea/setu-translate/tmp" --total_procs 24 --batch_size 350 --devices "0" --cache_dir "${cache_dir}/translate/cache"
    echo "Translation completed successfully." >> process_log.txt

    # Decode
    echo "Starting decoding..." >> process_log.txt
    python ${repo_base_path}/stages/decode.py --data_files "${base_save_dir}/model_out/*/*.arrow" --decode_dir "${base_save_dir}/decode" --format arrow --batch_size 2084 --total_procs 24 --src_lang eng_Latn --tgt_lang pan_Guru --cache_dir "${cache_dir}/decode/cache"
    echo "Decoding completed successfully." >> process_log.txt

    # Replace
    echo "Starting replace operation..." >> process_log.txt
    python ${repo_base_path}/stages/replace.py --paths_data "${base_save_dir}/templated/*.arrow" --batch_size 2084 --num_procs 24 --translated_save_path "/mnt/sea/setu_output/${source_type}/translated_${file_name_without_extension}" --cache_dir "${cache_dir}/replace/cache"
    echo "Replace operation completed successfully." >> process_log.txt

    # Log end time
    echo "Process completed at $(date)" >> process_log.txt

    # If the script completes without errors, disable the trap to not hold the terminal open unnecessarily
    trap - ERR

    echo "Cleaning up cache directories... This can take a file because of large number of files created during the process. Please wait..."
    echo "Deleting Cache..."
    for dir in /mnt/sea/setu_cache/*; do
        if [ -d "$dir" ]; then  # Check if it is a directory
            rsync -a --delete /mnt/sea/dolma/empty_dir/ "$dir/"
        fi
    done
    echo "Deleting Intermediate saved datasets..."
    for dir in /mnt/sea/setu/*; do
        if [ -d "$dir" ]; then  # Check if it is a directory
            rsync -a --delete /mnt/sea/dolma/empty_dir/ "$dir/"
        fi
    done

    echo "Shard processing completed successfully. Moving to next shard..."
    echo "----------------------------------------------" >> process_log.txt
    echo "Shard index ${file_name_without_extension} processing completed successfully. Moving to next shard..." >> process_log.txt

done

echo "Process completed successfully. Exiting..."

