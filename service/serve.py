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

WINDOW_LENGTH = 512

ds_config_path = "../examples/ds_config.json"
with open (ds_config_path, "r") as f:
    ds_config = json.load(f)

model_name_or_path = '../output_models/gpt_neo2.7B_inst_tuning/'
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

    return text_out

if __name__ == "__main__":
    # Open the file using jsonlines library
    with jsonlines.open(sys.argv[1]) as reader:
        # Read the first ten lines and extract the "text" field from each line
        lines = list(reader)
        if "test_input" in lines[0]:
            inputs = [line['test_input'] for i, line in enumerate(lines)]
        else:
            inputs = [line['text'] for i, line in enumerate(lines)]

    # Process the inputs in batches
    for idx, text in enumerate(tqdm.tqdm(inputs)):
        response = inference_one(text)

        # Write the batch responses to an output file
        with jsonlines.open(sys.argv[2], mode='a') as writer:
            writer.write({'test_output': response, **lines[idx]})