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

ds_config_path = "../examples/ds_config.json"
with open (ds_config_path, "r") as f:
    ds_config = json.load(f)

model_name_or_path = 'OptimalScale/gpt-neo2.7B-inst-tuning'
model_args = ModelArguments(model_name_or_path=model_name_or_path)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)


def inference_one(prompt):
    if prompt == "":
        return ""
    prompt = f"""Input: User{prompt}\n Assistant:"""
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

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file_base = sys.argv[2]
    worker_index = int(sys.argv[3])
    num_workers = int(sys.argv[4])

    with jsonlines.open(input_file) as reader:
        lines = list(reader)
        total_lines = len(lines)
        shard_size = total_lines // num_workers
        shard_start = shard_size * worker_index
        shard_end = total_lines if worker_index == num_workers - 1 else shard_start + shard_size
        shard = lines[shard_start:shard_end]

        if "test_input" in shard[0]:
            inputs = [line['test_input'] for i, line in enumerate(shard)]
        else:
            inputs = [line['text'] for i, line in enumerate(shard)]

    output_file = f"{output_file_base}_shard_{worker_index}.jsonl"
    for idx, text in enumerate(tqdm.tqdm(inputs)):
        response = inference_one(text)
        with jsonlines.open(output_file, mode='a') as writer:
            writer.write({'test_output': response, **shard[idx]})