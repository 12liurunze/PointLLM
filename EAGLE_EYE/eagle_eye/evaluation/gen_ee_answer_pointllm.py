import argparse
import json
import os
import re
import sys
import time

import torch

from eagle_eye.model.point_ee_model import PointEeModel


def add_pointllm_to_path(pointllm_repo_path):
    if pointllm_repo_path and pointllm_repo_path not in sys.path:
        sys.path.insert(0, pointllm_repo_path)


def load_pointllm_helpers(pointllm_repo_path):
    add_pointllm_to_path(pointllm_repo_path)
    from pointllm.conversation import SeparatorStyle, conv_templates
    from pointllm.data import load_objaverse_point_cloud
    from pointllm.model.utils import KeywordsStoppingCriteria

    return conv_templates, SeparatorStyle, load_objaverse_point_cloud, KeywordsStoppingCriteria


def build_point_prompt(model, model_path, question):
    point_backbone_config = model.base_model.get_model().point_backbone_config
    point_token_len = point_backbone_config["point_token_len"]
    patch = point_backbone_config["default_point_patch_token"]
    start = point_backbone_config.get("default_point_start_token")
    end = point_backbone_config.get("default_point_end_token")

    if getattr(model.base_model.config, "mm_use_point_start_end", False):
        question = start + patch * point_token_len + end + "\n" + question
    else:
        question = patch * point_token_len + "\n" + question

    if "v1" not in model_path.lower():
        raise NotImplementedError("Only PointLLM Vicuna v1-style conversation templates are supported.")

    return question, "vicuna_v1_1"


def run_one(args, model, tokenizer, helpers, sample):
    conv_templates, SeparatorStyle, load_objaverse_point_cloud, KeywordsStoppingCriteria = helpers

    object_id = sample["object_id"]
    question = sample.get("question", args.question)
    point_cloud = load_objaverse_point_cloud(
        args.data_path, object_id, pointnum=args.pointnum, use_color=True
    )
    point_clouds = torch.from_numpy(point_cloud).unsqueeze(0).to(
        device=model.base_model.device, dtype=args.torch_dtype
    )

    question, conv_mode = build_point_prompt(model, args.base_model_path, question)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids, device=model.base_model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    start_time = time.perf_counter()
    output_ids = model.eagenerate(
        input_ids=input_ids,
        point_clouds=point_clouds,
        stopping_criteria=[stopping_criteria],
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    elapsed = time.perf_counter() - start_time

    input_token_len = input_ids.shape[1]
    output = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0].strip()
    stop_pos = output.find(stop_str) if stop_str else -1
    if stop_pos >= 0:
        output = output[:stop_pos].strip()

    return {
        "object_id": object_id,
        "question": sample.get("question", args.question),
        "answer": output,
        "time": elapsed,
        "new_tokens": int(output_ids.shape[1] - input_token_len),
    }


def load_samples(args):
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        end = min(args.end, len(data)) if args.end is not None and args.end >= 0 else len(data)
        for idx in range(args.start, end):
            sample = data[idx]
            conversations = sample.get("conversations", [])
            question = args.question
            for turn in conversations:
                if turn.get("from") == "human":
                    question = turn.get("value", question)
                    break
            question = re.sub(r"\s*<point>\s*", "", question).strip()
            yield {
                "object_id": sample["object_id"],
                "question": question,
                "sample_index": idx,
            }
        return

    if args.input_jsonl:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        yield {"object_id": args.object_id, "question": args.question}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--ee-model-path", required=True)
    parser.add_argument("--pointllm-repo-path", default=None)
    parser.add_argument("--point-backbone-config-name", default=None)
    parser.add_argument("--force-single-point-proj", action="store_true")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default="pointllm_ee_answers.jsonl")
    parser.add_argument("--question", default="Describe this 3D object in detail.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--torch-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default=None)
    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_mapping[args.torch_dtype]

    if args.input_json is None and args.input_jsonl is None and args.object_id is None:
        raise ValueError("Provide --input-json, --input-jsonl, or --object-id.")

    helpers = load_pointllm_helpers(args.pointllm_repo_path)

    model_kwargs = {
        "low_cpu_mem_usage": args.device_map is not None,
        "torch_dtype": args.torch_dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = PointEeModel.from_pretrained(
        base_model_path=args.base_model_path,
        ee_model_path=args.ee_model_path,
        pointllm_repo_path=args.pointllm_repo_path,
        point_backbone_config_name=args.point_backbone_config_name,
        force_single_point_proj=args.force_single_point_proj,
        **model_kwargs,
    )
    if args.device_map is None:
        model = model.cuda()
    model.eval()
    tokenizer = model.get_tokenizer()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for sample in load_samples(args):
            result = run_one(args, model, tokenizer, helpers, sample)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(result)


if __name__ == "__main__":
    main()
