import argparse
import gc
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer

from eagle_eye.model.point_ee_model import PointEeModel, load_checkpoint_tensors_by_prefix


def add_pointllm_to_path(pointllm_repo_path):
    if pointllm_repo_path and pointllm_repo_path not in sys.path:
        sys.path.insert(0, pointllm_repo_path)


def load_pointllm_helpers(pointllm_repo_path):
    add_pointllm_to_path(pointllm_repo_path)
    from pointllm.conversation import SeparatorStyle, conv_templates
    from pointllm.data import load_objaverse_point_cloud
    from pointllm.model.utils import KeywordsStoppingCriteria

    return conv_templates, SeparatorStyle, load_objaverse_point_cloud, KeywordsStoppingCriteria


def load_samples(args):
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        end = min(args.end, len(data)) if args.end is not None and args.end >= 0 else len(data)
        for idx in range(args.start, end):
            sample = data[idx]
            conversations = sample.get("conversations", [])
            question = args.question
            answer = None
            for turn in conversations:
                if turn.get("from") == "human":
                    question = turn.get("value", question)
                elif turn.get("from") == "gpt" and answer is None:
                    answer = turn.get("value")
            question = re.sub(r"\s*<point>\s*", "", question).strip()
            yield {
                "object_id": sample["object_id"],
                "question": question,
                "ground_truth": answer,
                "sample_index": idx,
            }
        return

    if args.input_jsonl:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return

    yield {"object_id": args.object_id, "question": args.question, "ground_truth": None}


def build_prompt(model, model_path, tokenizer, helpers, question):
    conv_templates, SeparatorStyle, _, KeywordsStoppingCriteria = helpers
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config["point_token_len"]
    patch = point_backbone_config["default_point_patch_token"]
    start = point_backbone_config.get("default_point_start_token")
    end = point_backbone_config.get("default_point_end_token")

    if getattr(model.config, "mm_use_point_start_end", False):
        question = start + patch * point_token_len + end + "\n" + question
    else:
        question = patch * point_token_len + "\n" + question

    if "v1" not in model_path.lower():
        raise NotImplementedError("Only PointLLM Vicuna v1-style conversation templates are supported.")

    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = torch.as_tensor(tokenizer([conv.get_prompt()]).input_ids, device=model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    return input_ids, stop_str, stopping_criteria


def load_point_cloud(args, helpers, model, sample):
    _, _, load_objaverse_point_cloud, _ = helpers
    point_cloud = load_objaverse_point_cloud(
        args.data_path, sample["object_id"], pointnum=args.pointnum, use_color=True
    )
    return torch.from_numpy(point_cloud).unsqueeze(0).to(device=model.device, dtype=args.torch_dtype)


def decode_new_tokens(tokenizer, output_ids, input_len, stop_str):
    output = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
    stop_pos = output.find(stop_str) if stop_str else -1
    if stop_pos >= 0:
        output = output[:stop_pos].strip()
    return output


def maybe_patch_single_point_projector(args):
    if not args.force_single_point_proj:
        return
    import pointllm.model.pointllm as pointllm_modeling

    original_cfg_from_yaml_file = pointllm_modeling.cfg_from_yaml_file

    def cfg_from_yaml_file_single_proj(*cfg_args, **cfg_kwargs):
        point_bert_config = original_cfg_from_yaml_file(*cfg_args, **cfg_kwargs)
        point_bert_config.model.projection_hidden_layer = 0
        if "projection_hidden_dim" in point_bert_config.model:
            del point_bert_config.model.projection_hidden_dim
        return point_bert_config

    pointllm_modeling.cfg_from_yaml_file = cfg_from_yaml_file_single_proj


def load_baseline_model(args):
    add_pointllm_to_path(args.pointllm_repo_path)
    from pointllm.model import PointLLMLlamaForCausalLM
    from pointllm.utils import disable_torch_init

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    config = AutoConfig.from_pretrained(args.base_model_path)
    if args.point_backbone_config_name:
        config.point_backbone_config_name = args.point_backbone_config_name
    if args.force_single_point_proj:
        config.force_single_point_proj = True
        maybe_patch_single_point_projector(args)

    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.base_model_path,
        config=config,
        low_cpu_mem_usage=False,
        torch_dtype=args.torch_dtype,
    )
    load_checkpoint_tensors_by_prefix(model, args.base_model_path, prefixes=("model.point_proj.",))
    model = model.cuda()
    model.config.use_cache = True
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()
    return model, tokenizer


def run_baseline(args, samples, helpers):
    model, tokenizer = load_baseline_model(args)
    results = []
    skipped = []
    for sample in samples:
        try:
            point_clouds = load_point_cloud(args, helpers, model, sample)
            input_ids, stop_str, stopping_criteria = build_prompt(
                model, args.base_model_path, tokenizer, helpers, sample["question"]
            )
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=args.temperature > 1e-5,
                    temperature=max(args.temperature, 1.0),
                    top_k=args.top_k,
                    top_p=args.top_p if args.top_p > 0 else 1.0,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            input_len = input_ids.shape[1]
            answer = decode_new_tokens(tokenizer, output_ids, input_len, stop_str)
            results.append(
                {
                    **sample,
                    "baseline_answer": answer,
                    "baseline_time": elapsed,
                    "baseline_new_tokens": int(output_ids.shape[1] - input_len),
                }
            )
            print({"mode": "baseline", **results[-1]})
        except Exception as exc:
            item = {
                **sample,
                "stage": "baseline",
                "error": f"{type(exc).__name__}: {exc}",
            }
            skipped.append(item)
            print({"mode": "skip", **item})

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results, skipped


def load_eagle_model(args):
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
    return model, model.get_tokenizer()


def run_eagle(args, baseline_results, helpers):
    model, tokenizer = load_eagle_model(args)
    results = []
    skipped = []
    for sample in baseline_results:
        try:
            point_clouds = load_point_cloud(args, helpers, model.base_model, sample)
            input_ids, stop_str, stopping_criteria = build_prompt(
                model.base_model, args.base_model_path, tokenizer, helpers, sample["question"]
            )
            torch.cuda.synchronize()
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
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            input_len = input_ids.shape[1]
            answer = decode_new_tokens(tokenizer, output_ids, input_len, stop_str)
            result = {
                **sample,
                "eagle_answer": answer,
                "eagle_time": elapsed,
                "eagle_new_tokens": int(output_ids.shape[1] - input_len),
            }
            baseline_time = result["baseline_time"]
            result["speedup"] = baseline_time / elapsed if elapsed > 0 else None
            results.append(result)
            print({"mode": "eagle", **result})
        except Exception as exc:
            item = {
                **sample,
                "stage": "eagle",
                "error": f"{type(exc).__name__}: {exc}",
            }
            skipped.append(item)
            print({"mode": "skip", **item})
    return results, skipped


def summarize(results, skipped):
    total_baseline_time = sum(item["baseline_time"] for item in results)
    total_eagle_time = sum(item["eagle_time"] for item in results)
    total_baseline_tokens = sum(item["baseline_new_tokens"] for item in results)
    total_eagle_tokens = sum(item["eagle_new_tokens"] for item in results)
    return {
        "num_samples": len(results),
        "num_skipped": len(skipped),
        "baseline_total_time": total_baseline_time,
        "eagle_total_time": total_eagle_time,
        "speedup_by_total_time": total_baseline_time / total_eagle_time
        if total_eagle_time > 0
        else None,
        "baseline_tokens_per_second": total_baseline_tokens / total_baseline_time
        if total_baseline_time > 0
        else None,
        "eagle_tokens_per_second": total_eagle_tokens / total_eagle_time
        if total_eagle_time > 0
        else None,
        "baseline_total_new_tokens": total_baseline_tokens,
        "eagle_total_new_tokens": total_eagle_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--ee-model-path", required=True)
    parser.add_argument("--pointllm-repo-path", default=None)
    parser.add_argument("--point-backbone-config-name", default="PointTransformer_8192point_2layer")
    parser.add_argument("--force-single-point-proj", action="store_true")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--question", default="Describe this 3D object in detail.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    samples = list(load_samples(args))
    if not samples:
        raise ValueError("No samples to compare.")

    baseline_results, baseline_skipped = run_baseline(args, samples, helpers)
    results, eagle_skipped = run_eagle(args, baseline_results, helpers)
    skipped = baseline_skipped + eagle_skipped
    summary = summarize(results, skipped)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    skipped_jsonl = args.output_jsonl + ".skipped"
    with open(skipped_jsonl, "w", encoding="utf-8") as f:
        for item in skipped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print({"summary": summary})


if __name__ == "__main__":
    main()
