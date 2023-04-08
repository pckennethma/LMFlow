import argparse
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from lmflow.models.hf_decoder_model import HFDecoderModel
import torch.distributed as dist
from transformers import HfArgumentParser
import io
import json
import torch
import os, jsonlines, tqdm, sys


def inference_one(prompt):
    if prompt == "":
        return ""
    prompt = f"""Input: User: {prompt}\n Assistant:"""
    inputs = model.encode(prompt, return_tensors="pt").to(device=local_rank)
    outputs = model.inference(inputs, max_new_tokens=250,temperature=0.9, do_sample=False)
    text_out = model.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
    try:
        text_out = text_out[prompt_length:].strip("\n").split("\n\nDefintion:")[0]
    except:
        text_out = text_out[prompt_length:].strip("\n")
    try:
        text_out = text_out.split("User:")[0]
    except:
        pass

    return text_out



def inference_many(prompt):
    model.tokenizer.pad_token = model.tokenizer.eos_token_id
    model.tokenizer.padding_side='left'
    
    
    if prompt == "":
        return ""
    
    prompt_tmp = []
    for i in prompt:
        prompt_tmp.append(f"""Input: User{i}\n Assistant:""")
        
    prompt = prompt_tmp
    
    
    with torch.no_grad():
    
        prompt_all = model.tokenizer.batch_encode_plus(tuple(prompt), pad_to_max_length=True)

        prompt_all = torch.tensor(prompt_all['input_ids']).cuda()
    
            
        outputs = model.inference(prompt_all, max_new_tokens=250,temperature=0.9, do_sample=False)
        text_out_final = []
        for i in range(len(outputs)):
            text_out = model.decode(outputs[i], skip_special_tokens=True)

            prompt_length = len(model.decode(prompt_all[i], skip_special_tokens=True,))
            try:
                text_out = text_out[prompt_length:].strip("\n").split("\n\nDefintion:")[0]
            except:
                text_out = text_out[prompt_length:].strip("\n")
            try:
                text_out = text_out.split("User:")[0]
            except:
                pass
            text_out_final.append(text_out)
        
    
    return text_out_final

if __name__ == "__main__":
    # export CUDA_VISIBLE_DEVICES=3; deepspeed  --master_port  $(echo $((29500 + CUDA_VISIBLE_DEVICES))) serve_shard.py ~/reflection_explain_input_1k.jsonl ~/reflection_explain_output_1k.jsonl $CUDA_VISIBLE_DEVICES 4
    parser = argparse.ArgumentParser(description="Process a shard of a JSONL file with an AI model.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("output_file_base", type=str, help="Base name of the output JSONL file")
    parser.add_argument("worker_index", type=int, help="Index of the current worker")
    parser.add_argument("num_workers", type=int, help="Total number of workers")
    parser.add_argument("batch_size", type=int, help="batch size for infering")

    args, _ = parser.parse_known_args()
    
    batch_size = args.batch_size

    with jsonlines.open(args.input_file) as reader:
        lines = list(reader)
        total_lines = len(lines)
        shard_size = total_lines // args.num_workers
        shard_start = shard_size * args.worker_index
        shard_end = total_lines if args.worker_index == args.num_workers - 1 else shard_start + shard_size
        shard = lines[shard_start:shard_end]

        if os.path.exists(f"{args.output_file_base}_shard_{args.worker_index}.jsonl"):
            with jsonlines.open(f"{args.output_file_base}_shard_{args.worker_index}.jsonl") as reader:
                existing_lines = list(reader)
                if "test_input" in shard[0]:
                    existing_inputs = set([line['test_input'] for i, line in enumerate(existing_lines)])
                    inputs = [line['test_input'] for i, line in enumerate(shard) if line['test_input'] not in existing_inputs]
                else:
                    existing_inputs = set([line['text'] for i, line in enumerate(existing_lines)])
                    inputs = [line['text'] for i, line in enumerate(shard) if line['text'] not in existing_inputs]
        else:
            if "test_input" in shard[0]:
                inputs = [line['test_input'] for i, line in enumerate(shard)]
            else:
                inputs = [line['text'] for i, line in enumerate(shard)]
    if len(inputs) == 0:
        sys.exit(0)


    ds_config_path = "../examples/ds_config.json"
    with open (ds_config_path, "r") as f:
        ds_config = json.load(f)

    model_name_or_path = 'OptimalScale/gpt-neo2.7B-inst-tuning'
#     model_name_or_path = "decapoda-research/llama-13b-hf"
#     lora_path = "../output_models/llama13b-lora-170k"
    
    
#     model_name_or_path = "decapoda-research/llama-7b-hf"
#     lora_path = "../output_models/llama7b-lora-170k"
    
    try:
        model_args = ModelArguments(model_name_or_path=model_name_or_path, lora_model_path=lora_path)
    except:
        model_args = ModelArguments(model_name_or_path=model_name_or_path)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)
    
    
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    
    print("start infer all cases")
    batch_size = args.batch_size
    total_ans = []


    if batch_size == 1:
        output_file = f"{args.output_file_base}_shard_{args.worker_index}.jsonl"
        for idx, text in enumerate(tqdm.tqdm(inputs)):
            response = inference_one(text)
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write({'test_output': response, **shard[idx]})
    else:
        print("start multi bs infer all cases")
        for cur in tqdm.tqdm(batch(inputs,batch_size),total=len(inputs)//batch_size):
            total_ans.extend(inference_many(cur))

        
        output_file = f"{args.output_file_base}_shard_{args.worker_index}.jsonl"
        for idx, text in enumerate(tqdm.tqdm(inputs)):
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write({'test_output': total_ans[idx], **shard[idx]})
                
                
                
