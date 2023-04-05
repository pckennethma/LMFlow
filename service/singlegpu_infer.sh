#!/bin/bash

NVIDIA_VISIBLE_DEVICES=1 deepspeed --num_gpus 1 serve.py full_benchmark.jsonl full_benchmark_2.7b_out.jsonl.jsonl 10
