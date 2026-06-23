import json
import os
import time
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import torch

from eagle_eye.evaluation.compare_pointllm_eagle import (
    build_prompt,
    decode_new_tokens,
    load_eagle_model,
    load_point_cloud,
    load_pointllm_helpers,
)
from eagle_eye.model import choices as tree_choice_defs


BASELINE_JSONL = "/root/autodl-tmp/pointllm_compare_eagle_compress_nofps_20260602_200451.jsonl"
OUT_PATH = "/root/autodl-tmp/pointllm_octree_ab_20260609.json"

TREE35 = tree_choice_defs.mc_sim_7b_63 + [
    [4], [5],
    [0, 3], [1, 2], [2, 2], [3, 1], [4, 0],
    [1, 0, 1], [1, 1, 0], [2, 0, 0],
]
TREE45 = TREE35 + [
    [6], [7],
    [0, 4], [1, 3], [2, 3], [3, 2], [4, 1], [5, 0],
    [0, 0, 3], [0, 1, 2],
]
TREE80 = (
    [[a] for a in range(10)]
    + [[a, b] for a in range(8) for b in range(5)]
    + [[a, b, c] for a in range(4) for b in range(3) for c in range(2)]
    + [[0] * depth for depth in range(4, 10)]
)
LEARNED_TREE35 = [
    [0], [0, 0], [1], [0, 0, 0], [0, 1], [2], [1, 0], [0, 2], [3],
    [0, 0, 1], [0, 1, 0], [0, 3], [0, 0, 0, 0], [5], [0, 4],
    [0, 2, 0], [2, 0], [4], [1, 0, 0], [3, 0], [1, 1], [7], [8],
    [5, 0], [0, 0, 0, 0, 0], [2, 0, 0], [6], [9], [1, 2],
    [1, 0, 1], [1, 3], [2, 1], [4, 0], [0, 1, 1], [6, 0],
]
LEARNED_TREE45 = LEARNED_TREE35 + [
    [0, 2, 1], [3, 0, 0], [1, 4], [3, 1], [7, 0],
    [1, 1, 0], [2, 0, 1], [2, 3], [2, 4], [1, 2, 0],
]


def load_rows(limit):
    offset = int(os.environ.get("BENCH_OFFSET", "0"))
    rows = []
    seen = 0
    with open(BASELINE_JSONL, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                if seen < offset:
                    seen += 1
                    continue
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


@torch.no_grad()
def run_condition(args, model, tokenizer, helpers, rows, condition):
    for key in (
        "POINT_ADAPTIVE_TREE",
        "POINT_TARGET_KEEP_RATIO",
        "POINT_TARGET_SUMMARY_COUNT",
    ):
        if key not in condition["env"]:
            os.environ.pop(key, None)
    for key, value in condition["env"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)

    elapsed_total = 0.0
    accepted_total = 0
    steps_total = 0
    eagle_tokens = 0
    exact_matches = 0
    text_similarity_total = 0.0
    profile_total = {}
    samples = []
    for index, sample in enumerate(rows):
        point_clouds = load_point_cloud(args, helpers, model.base_model, sample)
        input_ids, stop_str, stopping_criteria = build_prompt(
            model.base_model, args.base_model_path, tokenizer, helpers, sample["question"]
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        output_ids, stats = model.eagenerate(
            input_ids=input_ids,
            point_clouds=point_clouds,
            stopping_criteria=[stopping_criteria],
            temperature=0.0,
            max_new_tokens=128,
            max_length=2048,
            tree_choices=condition.get("tree_choices", tree_choice_defs.mc_sim_7b_63),
            return_stats=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        answer = decode_new_tokens(tokenizer, output_ids, input_ids.shape[1], stop_str)
        new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
        elapsed_total += elapsed
        accepted_total += stats["accepted_tokens_total"]
        steps_total += stats["accept_steps"]
        eagle_tokens += new_tokens
        exact_matches += int(answer == sample["baseline_answer"])
        text_similarity = SequenceMatcher(
            None, sample["baseline_answer"], answer, autojunk=False
        ).ratio()
        text_similarity_total += text_similarity
        for key, value in (stats.get("profile") or {}).items():
            profile_total[key] = profile_total.get(key, 0.0) + value
        samples.append(
            {
                "object_id": sample["object_id"],
                "time": elapsed,
                "new_tokens": new_tokens,
                "mat": stats["mat"],
                "exact_match": answer == sample["baseline_answer"],
                "text_similarity": text_similarity,
                "compression": getattr(model, "last_point_token_compression", None),
                "adaptive_tree_counts": stats.get("adaptive_tree_counts"),
                "accepted_rank_paths": stats.get("accepted_rank_paths"),
                "spatial_complexity": stats.get("spatial_complexity"),
            }
        )
        if (index + 1) % 10 == 0:
            print(condition["name"], index + 1, flush=True)

    baseline_time = sum(row["baseline_time"] for row in rows)
    baseline_tokens = sum(row["baseline_new_tokens"] for row in rows)
    return {
        "name": condition["name"],
        "tree_nodes": len(condition.get("tree_choices", tree_choice_defs.mc_sim_7b_63)),
        "env": condition["env"],
        "num_samples": len(rows),
        "baseline_total_time": baseline_time,
        "eagle_total_time": elapsed_total,
        "speedup": baseline_time / elapsed_total,
        "mat": accepted_total / steps_total,
        "accepted_tokens_total": accepted_total,
        "accept_steps": steps_total,
        "baseline_new_tokens": baseline_tokens,
        "eagle_new_tokens": eagle_tokens,
        "baseline_tokens_per_second": baseline_tokens / baseline_time,
        "eagle_tokens_per_second": eagle_tokens / elapsed_total,
        "throughput_speedup": (eagle_tokens / elapsed_total) / (baseline_tokens / baseline_time),
        "exact_match_rate": exact_matches / len(rows),
        "mean_text_similarity": text_similarity_total / len(rows),
        "profile": profile_total or None,
        "adaptive_tree_counts": {
            key: sum(
                (sample.get("adaptive_tree_counts") or {}).get(key, 0)
                for sample in samples
            )
            for key in ("narrow", "medium", "wide")
        }
        if any(sample.get("adaptive_tree_counts") for sample in samples)
        else None,
        "samples": samples,
    }


def main():
    args = SimpleNamespace(
        base_model_path="/root/autodl-tmp/point7B_v1.1",
        ee_model_path=os.environ.get(
            "BENCH_HEAD",
            "/root/autodl-tmp/pointllm_eagle_head_compress_fps4096_0p5_8_20260602_203738",
        ),
        pointllm_repo_path="/root/autodl-tmp/pointLLM",
        point_backbone_config_name="PointTransformer_8192point_2layer",
        force_single_point_proj=False,
        device_map=None,
        torch_dtype=torch.float16,
        data_path="/root/autodl-tmp/pointLLM/data/objaverse_data",
        pointnum=8192,
    )
    rows = load_rows(int(os.environ.get("BENCH_LIMIT", "40")))
    helpers = load_pointllm_helpers(args.pointllm_repo_path)
    model, tokenizer = load_eagle_model(args)

    conditions = [
        {
            "name": "learned_semantic_tree35",
            "tree_choices": LEARNED_TREE35,
            "env": {
                "POINT_TARGET_KEEP_RATIO": None,
                "POINT_TARGET_SUMMARY_COUNT": None,
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "learned_semantic_tree45",
            "tree_choices": LEARNED_TREE45,
            "env": {
                "POINT_TARGET_KEEP_RATIO": None,
                "POINT_TARGET_SUMMARY_COUNT": None,
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "collector_tree80",
            "tree_choices": TREE80,
            "env": {
                "POINT_TARGET_KEEP_RATIO": None,
                "POINT_TARGET_SUMMARY_COUNT": None,
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "adaptive_semantic_tree45",
            "tree_choices": TREE45,
            "env": {
                "POINT_ADAPTIVE_TREE": "1",
                "POINT_ADAPTIVE_NARROW_ACCEPT": "3",
                "POINT_ADAPTIVE_MEDIUM_ACCEPT": "1",
                "POINT_ADAPTIVE_MARGIN": "0.8",
                "POINT_ADAPTIVE_COMPLEXITY_THRESHOLD": "0.65",
                "POINT_TARGET_KEEP_RATIO": None,
                "POINT_TARGET_SUMMARY_COUNT": None,
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_TOKEN_OCTREE_DEPTH": "4",
                "POINT_TOKEN_HIERARCHY_LEVELS": "1,2,4",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        *[
            {
                "name": f"target_shoss_tree{tree_label}_keep{ratio_label}",
                "tree_choices": tree_choices,
                "env": {
                    "POINT_TARGET_KEEP_RATIO": str(ratio),
                    "POINT_TARGET_SUMMARY_COUNT": "8",
                    "POINT_TARGET_OCTREE_DEPTH": "4",
                    "POINT_SPATIAL_NUM_POINTS": "0",
                    "POINT_TOKEN_KEEP_RATIO": "0.5",
                    "POINT_TOKEN_SUMMARY_COUNT": "8",
                    "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                    "POINT_TOKEN_SPATIAL_MODE": "none",
                    "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
                },
            }
            for tree_label, tree_choices in [("35", TREE35), ("45", TREE45)]
            for ratio_label, ratio in [("12p5", 0.125), ("25", 0.25), ("50", 0.5)]
        ],
        *[
            {
                "name": f"hoss_tree45_keep{label}",
                "tree_choices": TREE45,
                "env": {
                    "POINT_SPATIAL_NUM_POINTS": "0",
                    "POINT_TOKEN_KEEP_RATIO": str(ratio),
                    "POINT_TOKEN_SUMMARY_COUNT": "8",
                    "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                    "POINT_TOKEN_SPATIAL_MODE": "hierarchical_octree",
                    "POINT_TOKEN_OCTREE_DEPTH": "4",
                    "POINT_TOKEN_HIERARCHY_LEVELS": "1,2,4",
                    "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
                },
            }
            for label, ratio in [
                ("3p125", 0.03125),
                ("6p25", 0.0625),
                ("12p5", 0.125),
                ("25", 0.25),
                ("50", 0.5),
            ]
        ],
        {
            "name": "tree35_trainmatch",
            "tree_choices": TREE35,
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "tree45_trainmatch",
            "tree_choices": TREE45,
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "tree5_linear_trainmatch",
            "tree_choices": [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "tree9_trainmatch",
            "tree_choices": tree_choice_defs.simple_depth5,
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "tree15_trainmatch",
            "tree_choices": [
                [0], [1], [2],
                [0, 0], [0, 1], [1, 0], [1, 1],
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                [0, 0, 0, 0], [0, 0, 0, 1],
                [0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
            ],
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "tree25_trainmatch",
            "tree_choices": tree_choice_defs.mc_sim_7b_63,
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "legacy_recompute_trainmatch_keep50_sum8",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": "1",
            },
        },
        {
            "name": "aligned_trainmatch_keep50_sum8",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "octree_trainmatch_keep50_sum8",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "0.5",
                "POINT_TOKEN_SPATIAL_MODE": "octree",
                "POINT_TOKEN_OCTREE_DEPTH": "4",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "legacy_recompute_topk_keep25_sum0",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.25",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": "1",
            },
        },
        {
            "name": "topk_keep25_sum0",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.25",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "octree_keep25_sum8",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.25",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "octree",
                "POINT_TOKEN_OCTREE_DEPTH": "4",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "octree_keep25_sum16",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.25",
                "POINT_TOKEN_SUMMARY_COUNT": "16",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "octree",
                "POINT_TOKEN_OCTREE_DEPTH": "4",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "octree_keep50_sum8",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.5",
                "POINT_TOKEN_SUMMARY_COUNT": "8",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "octree",
                "POINT_TOKEN_OCTREE_DEPTH": "4",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "limit_keep12p5_sum0",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.125",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "limit_keep6p25_sum0",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.0625",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "limit_keep3p125_sum0",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.03125",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "limit_keep1token_sum0",
            "env": {
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "0.001",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_TOKEN_TEXT_WEIGHT": "1.0",
                "POINT_TOKEN_SPATIAL_MODE": "none",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        *[
            {
                "name": f"qhostc_fixed_{label}",
                "tree_choices": TREE45,
                "env": {
                    "POINT_TARGET_KEEP_RATIO": str(ratio),
                    "POINT_TARGET_SUMMARY_COUNT": str(summary_count),
                    "POINT_TARGET_ADAPTIVE_RATIO": None,
                    "POINT_TARGET_SEMANTIC_WEIGHT": "0.75",
                    "POINT_TARGET_DENSITY_WEIGHT": "0.15",
                    "POINT_TARGET_SEMANTIC_TEMPERATURE": "0.25",
                    "POINT_TARGET_HIERARCHY_LEVELS": "1,2,4",
                    "POINT_TARGET_OCTREE_DEPTH": "4",
                    "POINT_SPATIAL_NUM_POINTS": "0",
                    "POINT_TOKEN_KEEP_RATIO": "1.0",
                    "POINT_TOKEN_SUMMARY_COUNT": "0",
                    "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
                },
            }
            for label, ratio, summary_count in [
                ("12p5_sum8", 0.125, 8),
                ("25_sum8", 0.25, 8),
                ("50_sum8", 0.5, 8),
            ]
        ],
        {
            "name": "qhostc_adaptive_conservative",
            "tree_choices": TREE45,
            "env": {
                "POINT_TARGET_KEEP_RATIO": "0.25",
                "POINT_TARGET_SUMMARY_COUNT": "8",
                "POINT_TARGET_ADAPTIVE_RATIO": "1",
                "POINT_TARGET_SIMPLE_RATIO": "0.125",
                "POINT_TARGET_MEDIUM_RATIO": "0.25",
                "POINT_TARGET_COMPLEX_RATIO": "0.5",
                "POINT_TARGET_COMPLEXITY_LOW": "0.34",
                "POINT_TARGET_COMPLEXITY_HIGH": "0.52",
                "POINT_TARGET_SEMANTIC_WEIGHT": "0.75",
                "POINT_TARGET_DENSITY_WEIGHT": "0.15",
                "POINT_TARGET_SEMANTIC_TEMPERATURE": "0.25",
                "POINT_TARGET_HIERARCHY_LEVELS": "1,2,4",
                "POINT_TARGET_OCTREE_DEPTH": "4",
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "1.0",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
        {
            "name": "qhostc_adaptive_aggressive",
            "tree_choices": TREE45,
            "env": {
                "POINT_TARGET_KEEP_RATIO": "0.125",
                "POINT_TARGET_SUMMARY_COUNT": "8",
                "POINT_TARGET_ADAPTIVE_RATIO": "1",
                "POINT_TARGET_SIMPLE_RATIO": "0.0625",
                "POINT_TARGET_MEDIUM_RATIO": "0.125",
                "POINT_TARGET_COMPLEX_RATIO": "0.25",
                "POINT_TARGET_COMPLEXITY_LOW": "0.34",
                "POINT_TARGET_COMPLEXITY_HIGH": "0.52",
                "POINT_TARGET_SEMANTIC_WEIGHT": "0.75",
                "POINT_TARGET_DENSITY_WEIGHT": "0.15",
                "POINT_TARGET_SEMANTIC_TEMPERATURE": "0.25",
                "POINT_TARGET_HIERARCHY_LEVELS": "1,2,4",
                "POINT_TARGET_OCTREE_DEPTH": "4",
                "POINT_SPATIAL_NUM_POINTS": "0",
                "POINT_TOKEN_KEEP_RATIO": "1.0",
                "POINT_TOKEN_SUMMARY_COUNT": "0",
                "POINT_DRAFT_RECOMPUTE_HIDDEN": None,
            },
        },
    ]
    requested = {
        name.strip()
        for name in os.environ.get("BENCH_CONDITIONS", "").split(",")
        if name.strip()
    }
    if requested:
        conditions = [condition for condition in conditions if condition["name"] in requested]
    requested = os.environ.get("BENCH_CONDITIONS")
    if requested:
        names = set(requested.split(","))
        conditions = [condition for condition in conditions if condition["name"] in names]

    # Warm kernels and allocations without including the warmup in measurements.
    warmup_rows = rows[:1]
    warmup_condition = next(
        (condition for condition in conditions if not condition["env"].get("POINT_DRAFT_RECOMPUTE_HIDDEN")),
        conditions[0],
    )
    run_condition(args, model, tokenizer, helpers, warmup_rows, warmup_condition)

    results = []
    for condition in conditions:
        result = run_condition(args, model, tokenizer, helpers, rows, condition)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2), flush=True)
        Path(OUT_PATH).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
