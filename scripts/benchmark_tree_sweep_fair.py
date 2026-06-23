import gc
import json
import os
import time
from types import SimpleNamespace

import torch

from eagle_eye.evaluation.compare_pointllm_eagle import (
    build_prompt,
    load_eagle_model,
    load_point_cloud,
    load_pointllm_helpers,
)
from eagle_eye.model import choices as tree_choice_defs
from eagle_eye.model.kv_cache import initialize_past_key_values
from eagle_eye.model.utils import reset_tree_mode


BASELINE_JSONL = (
    "/root/autodl-tmp/pointllm_compare_eagle_compress_nofps_20260602_200451.jsonl"
)


@torch.inference_mode()
def main():
    args = SimpleNamespace(
        base_model_path="/root/autodl-tmp/point7B_v1.1",
        ee_model_path=os.environ.get(
            "BENCH_HEAD",
            "/root/autodl-tmp/pointllm_deterministic_margin_head_v1",
        ),
        pointllm_repo_path="/root/autodl-tmp/pointLLM",
        point_backbone_config_name="PointTransformer_8192point_2layer",
        force_single_point_proj=False,
        device_map=None,
        torch_dtype=torch.float16,
        data_path="/root/autodl-tmp/pointLLM/data/objaverse_data",
        pointnum=8192,
    )
    limit = int(os.environ.get("BENCH_LIMIT", "20"))
    rows = [
        json.loads(line)
        for line in open(BASELINE_JSONL, encoding="utf-8")
        if line.strip()
    ][:limit]
    tree_names = [
        value.strip()
        for value in os.environ.get(
            "BENCH_TREES",
            "pointllm_wide_35,pointllm_wide_45,pointllm_wide_55,pointllm_wide_65",
        ).split(",")
        if value.strip()
    ]
    trees = {name: getattr(tree_choice_defs, name) for name in tree_names}

    helpers = load_pointllm_helpers(args.pointllm_repo_path)
    model, tokenizer = load_eagle_model(args)
    shared_past, shared_data, shared_length = initialize_past_key_values(
        model.language_model
    )
    model.past_key_values = shared_past
    model.past_key_values_data = shared_data
    model.current_length_data = shared_length

    results = {
        name: {
            "tree_nodes": len(tree),
            "baseline_time": 0.0,
            "eagle_time": 0.0,
            "baseline_tokens": 0,
            "eagle_tokens": 0,
            "exact_samples": 0,
            "accepted_tokens": 0,
            "accept_steps": 0,
        }
        for name, tree in trees.items()
    }

    # Warm all tree buffers and kernels without counting the warmup.
    warm_sample = rows[0]
    warm_points = load_point_cloud(args, helpers, model.base_model, warm_sample)
    warm_ids, _, warm_stop = build_prompt(
        model.base_model,
        args.base_model_path,
        tokenizer,
        helpers,
        warm_sample["question"],
    )
    for name, tree in trees.items():
        model.eagenerate(
            input_ids=warm_ids,
            point_clouds=warm_points,
            stopping_criteria=[warm_stop],
            temperature=0.0,
            max_new_tokens=8,
            max_length=2048,
            tree_choices=tree,
            return_stats=True,
        )
        model.ee_layer.reset_kv()
    torch.cuda.synchronize()

    for sample_index, sample in enumerate(rows):
        point_clouds = load_point_cloud(args, helpers, model.base_model, sample)
        input_ids, _, baseline_stopping = build_prompt(
            model.base_model,
            args.base_model_path,
            tokenizer,
            helpers,
            sample["question"],
        )

        reset_tree_mode(model.language_model)
        shared_length.zero_()
        baseline_ids = input_ids.clone()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(128):
            if step == 0:
                outputs = model.base_model(
                    input_ids=baseline_ids,
                    point_clouds=point_clouds,
                    past_key_values=shared_past,
                    output_hidden_states=False,
                    num_logits_to_keep=1,
                )
            else:
                outputs = model.base_model(
                    input_ids=baseline_ids[:, -1:],
                    past_key_values=shared_past,
                    output_hidden_states=False,
                    num_logits_to_keep=1,
                )
            token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            baseline_ids = torch.cat((baseline_ids, token), dim=1)
            if tokenizer.eos_token_id is not None and int(token.item()) == tokenizer.eos_token_id:
                break
            verdict = baseline_stopping(baseline_ids, None)
            if bool(verdict.item()) if isinstance(verdict, torch.Tensor) else bool(verdict):
                break
        torch.cuda.synchronize()
        baseline_elapsed = time.perf_counter() - started
        baseline_new = baseline_ids[0, input_ids.shape[1] :].cpu()

        for name, tree in trees.items():
            _, _, eagle_stopping = build_prompt(
                model.base_model,
                args.base_model_path,
                tokenizer,
                helpers,
                sample["question"],
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            eagle_ids, stats = model.eagenerate(
                input_ids=input_ids,
                point_clouds=point_clouds,
                stopping_criteria=[eagle_stopping],
                temperature=0.0,
                max_new_tokens=128,
                max_length=2048,
                tree_choices=tree,
                return_stats=True,
            )
            torch.cuda.synchronize()
            eagle_elapsed = time.perf_counter() - started
            eagle_new = eagle_ids[0, input_ids.shape[1] :].cpu()
            result = results[name]
            result["baseline_time"] += baseline_elapsed
            result["eagle_time"] += eagle_elapsed
            result["baseline_tokens"] += int(baseline_new.numel())
            result["eagle_tokens"] += int(eagle_new.numel())
            result["exact_samples"] += int(torch.equal(baseline_new, eagle_new))
            result["accepted_tokens"] += stats["accepted_tokens_total"]
            result["accept_steps"] += stats["accept_steps"]
            model.ee_layer.reset_kv()
            del eagle_ids, eagle_new

        print(f"completed {sample_index + 1}/{len(rows)}", flush=True)
        del outputs, baseline_ids, baseline_new, point_clouds
        gc.collect()
        torch.cuda.empty_cache()

    for result in results.values():
        result["samples"] = len(rows)
        result["speedup"] = result["baseline_time"] / result["eagle_time"]
        result["throughput_speedup"] = (
            result["eagle_tokens"] / result["eagle_time"]
        ) / (result["baseline_tokens"] / result["baseline_time"])
        result["output_length_ratio"] = (
            result["eagle_tokens"] / result["baseline_tokens"]
        )
        result["token_exact_rate"] = result["exact_samples"] / len(rows)
        result["mat"] = result["accepted_tokens"] / result["accept_steps"]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
