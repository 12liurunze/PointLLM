import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import load_objaverse_point_cloud
from pointllm.eval.speculative_decode import speculative_decode, tree_speculative_decode


def load_point_cloud(data_path, object_id, dtype):
    point_cloud = load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=True)
    return torch.from_numpy(point_cloud).unsqueeze_(0).to(dtype).cuda()


def build_prompt(model, question):
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config["point_token_len"]
    default_point_patch_token = point_backbone_config["default_point_patch_token"]
    default_point_start_token = point_backbone_config["default_point_start_token"]
    default_point_end_token = point_backbone_config["default_point_end_token"]

    conv = conv_templates["vicuna_v1_1"].copy()
    qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + "\n" + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return prompt, stop_str


def run_vanilla(model, tokenizer, input_ids, point_clouds, max_length):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_length=max_length,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
    return output_ids, elapsed, new_tokens


def run_speculative(args, model, assistant_model, tokenizer, input_ids, point_clouds, max_length):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        if args.speculative_mode == "tree":
            output_ids, stats = tree_speculative_decode(
                model=model,
                assistant_model=assistant_model,
                input_ids=input_ids,
                point_clouds=point_clouds,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
                tree_depth=args.tree_depth,
                tree_branching_factor=args.tree_branching_factor,
                tree_max_paths=args.tree_max_paths,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                return_stats=True,
            )
        else:
            output_ids, stats = speculative_decode(
                model=model,
                assistant_model=assistant_model,
                input_ids=input_ids,
                point_clouds=point_clouds,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
                num_assistant_tokens=args.num_assistant_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                return_stats=True,
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
    return output_ids, elapsed, new_tokens, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--draft_model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/objaverse_data")
    parser.add_argument("--object_ids", type=str, nargs="+", required=True, help="List of Objaverse object ids.")
    parser.add_argument("--question", type=str, default="Please describe this object in detail.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--speculative_mode", type=str, default="linear", choices=["linear", "tree"])
    parser.add_argument("--num_assistant_tokens", type=int, default=8)
    parser.add_argument("--tree_depth", type=int, default=4)
    parser.add_argument("--tree_branching_factor", type=int, default=2)
    parser.add_argument("--tree_max_paths", type=int, default=16)
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save benchmark metrics in json.")
    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_mapping[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(args.model_name, use_cache=True, torch_dtype=torch_dtype).cuda().eval()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    assistant_model = PointLLMLlamaForCausalLM.from_pretrained(
        args.draft_model_name, use_cache=True, torch_dtype=torch_dtype
    ).cuda().eval()
    assistant_model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    aggregate = {
        "num_samples": 0,
        "vanilla_time": 0.0,
        "spec_time": 0.0,
        "vanilla_tokens": 0,
        "spec_tokens": 0,
        "draft_proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "target_corrections": 0,
    }

    for object_id in args.object_ids:
        point_clouds = load_point_cloud(args.data_path, object_id, torch_dtype)
        prompt, stop_str = build_prompt(model, args.question)
        input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()

        vanilla_output, vanilla_time, vanilla_tokens = run_vanilla(
            model, tokenizer, input_ids, point_clouds, max_length=args.max_length
        )
        spec_output, spec_time, spec_tokens, stats = run_speculative(
            args, model, assistant_model, tokenizer, input_ids, point_clouds, max_length=args.max_length
        )

        aggregate["num_samples"] += 1
        aggregate["vanilla_time"] += vanilla_time
        aggregate["spec_time"] += spec_time
        aggregate["vanilla_tokens"] += vanilla_tokens
        aggregate["spec_tokens"] += spec_tokens
        aggregate["draft_proposed_tokens"] += stats["draft_proposed_tokens"]
        aggregate["accepted_draft_tokens"] += stats["accepted_draft_tokens"]
        aggregate["target_corrections"] += stats["target_corrections"]

        vanilla_text = tokenizer.batch_decode(vanilla_output[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        spec_text = tokenizer.batch_decode(spec_output[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if vanilla_text.endswith(stop_str):
            vanilla_text = vanilla_text[:-len(stop_str)].strip()
        if spec_text.endswith(stop_str):
            spec_text = spec_text[:-len(stop_str)].strip()

        print(f"\n[Object] {object_id}")
        print(f"vanilla_time={vanilla_time:.3f}s, spec_time={spec_time:.3f}s, speedup={vanilla_time/max(spec_time,1e-6):.3f}x")
        print(f"acceptance_rate={stats['acceptance_rate']:.4f}, draft_proposed={stats['draft_proposed_tokens']}, accepted={stats['accepted_draft_tokens']}")
        print(f"[vanilla] {vanilla_text}")
        print(f"[spec]    {spec_text}")

    avg_vanilla_tps = aggregate["vanilla_tokens"] / max(aggregate["vanilla_time"], 1e-6)
    avg_spec_tps = aggregate["spec_tokens"] / max(aggregate["spec_time"], 1e-6)
    speedup = aggregate["vanilla_time"] / max(aggregate["spec_time"], 1e-6)
    acceptance_rate = aggregate["accepted_draft_tokens"] / max(aggregate["draft_proposed_tokens"], 1)

    summary = {
        "num_samples": aggregate["num_samples"],
        "speculative_mode": args.speculative_mode,
        "vanilla_total_time_sec": aggregate["vanilla_time"],
        "spec_total_time_sec": aggregate["spec_time"],
        "speedup_x": speedup,
        "vanilla_tokens_per_sec": avg_vanilla_tps,
        "spec_tokens_per_sec": avg_spec_tps,
        "draft_acceptance_rate": acceptance_rate,
        "draft_proposed_tokens": aggregate["draft_proposed_tokens"],
        "accepted_draft_tokens": aggregate["accepted_draft_tokens"],
        "target_corrections": aggregate["target_corrections"],
    }
    print("\n===== Benchmark Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.save_json is not None:
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Saved benchmark summary to {args.save_json}")


if __name__ == "__main__":
    main()
