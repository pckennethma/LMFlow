#!/bin/bash

# NVIDIA_VISIBLE_DEVICES=1 
# deepspeed --num_gpus 2 serve_shard_ds.py full_benchmark.jsonl full_benchmark_2.7b_out.jsonl 5

export CUDA_VISIBLE_DEVICES=1
deepspeed --master_port $((29500 + 2)) serve.py full_benchmark.jsonl full_benchmark_7b_out.jsonl 5

export CUDA_VISIBLE_DEVICES=0
deepspeed --master_port $((29500)) serve.py full_benchmark.jsonl full_benchmark_2.7b_out.jsonl 10

# wait

# python merge_shards.py full_benchmark_2.7b_out.jsonl 2
