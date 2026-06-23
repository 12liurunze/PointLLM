import argparse
import copy
import re
parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[1])
parser.add_argument(
    "--outdir", type=str, default="/home/dhz/llava-instruct-output/get_data_qwen2.5vl"
)
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoProcessor,
)
from transformers import Qwen2_5_VLForConditionalGeneration

from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template
from PIL import Image


bigname = "/home/dhz/Qwen2.5-VL-7B-Instruct"

data = "/home/dhz/LLaVA-Video-178K/0_30_s_academic_v0_1/academic_source"


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
    processor,
    split="train",
    select=None,
):
    ds = load_dataset(
        "json", data_files="/home/dhz/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json"
    )
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    #ds1 = ds.select(range(args.start, args.end))
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 1
    tokenizer = processor.tokenizer
    seen_ids = set()

    def preprocess_function_with_seen_ids(examples):
        return preprocess_function(examples, seen_ids)
    def preprocess_function(examples, seen_ids):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "pixel_values": [],
            "image_grid_thw": [],
            "loss_mask": [],
        }

        return new_examples

    ds1 = ds1.map(
        preprocess_function_with_seen_ids,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1

bigmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16,attn_implementation='eager')

bigprocessor = AutoProcessor.from_pretrained(bigname, use_fast=False)

ds = build_dataset_rank(bigprocessor)
print(ds)

bigmodel.eval()


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    pixel_values = data["pixel_values"]
    image_grid_thw = data["image_grid_thw"]
    attention_mask = torch.ones_like(input_ids)

    outs_big = bigmodel(
        input_ids=input_ids.cuda(),
        pixel_values=pixel_values.cuda(),
        attention_mask=attention_mask.cuda(),
        image_grid_thw=image_grid_thw.cuda(),
        output_hidden_states=True,
    )
    inputs_embeds = outs_big.hidden_states[0]
    hidden_state_big = outs_big.hidden_states[-1]


    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
    }
    return td


outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f"{name}/data_{idx}.ckpt")


for data in ds:
    outdata = ge(data)
    writedata(outdir, outdata)
