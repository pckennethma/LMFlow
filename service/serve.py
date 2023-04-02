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
import os

WINDOW_LENGTH = 512

ds_config_path = "../examples/ds_config.json"
with open (ds_config_path, "r") as f:
    ds_config = json.load(f)

model_name_or_path = '../output_models/gpt_neo2.7B_inst_tuning/'
# model_name_or_path = "gpt2"
# lora_path = '../output_models/instruction_ckpt/llama7b-lora/'
model_args = ModelArguments(model_name_or_path=model_name_or_path)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)


def inference(prompt):
    inputs = model.encode(prompt, return_tensors="pt").to(device=local_rank)
    outputs = model.inference(inputs, max_new_tokens=150,temperature=0.0, do_sample=False)
    text_out = model.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
    text_out = text_out[prompt_length:].strip("\n")
    return model


