#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file> <output_file> <num_shards> <num_batch>"
    exit 1
fi

input_file="$1"
output_file="$2"
num_shards="$3"
num_batch="$4"

# Run commands in parallel
for ((i=0; i<num_shards; i++)); do
    export CUDA_VISIBLE_DEVICES=$i
    deepspeed --master_port $((29500 + CUDA_VISIBLE_DEVICES)) serve_shard.py $input_file $output_file $CUDA_VISIBLE_DEVICES $num_shards $num_batch &
done

# Wait for all background processes to finish
wait

# Run the last command
python merge_shards.py $output_file $num_shards
