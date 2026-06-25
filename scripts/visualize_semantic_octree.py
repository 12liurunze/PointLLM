#!/usr/bin/env python
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer


def add_repo(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def equal_axes(ax, xyz):
    center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    radius = max((xyz.max(axis=0) - xyz.min(axis=0)).max() * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)


def scatter(ax, xyz, colors, size=2.0, alpha=0.9):
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=size,
        alpha=alpha,
        linewidths=0,
        depthshade=False,
    )
    equal_axes(ax, xyz)


def nearest_center_labels(points, centers, chunk=2048):
    labels = []
    centers = centers.float()
    for start in range(0, points.shape[0], chunk):
        cur = points[start : start + chunk].float()
        distances = torch.cdist(cur, centers)
        labels.append(distances.argmin(dim=1).cpu())
    return torch.cat(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--pointllm-repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=0.05)
    parser.add_argument("--summary-count", type=int, default=8)
    parser.add_argument("--octree-depth", type=int, default=4)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.25)
    args = parser.parse_args()

    add_repo(args.pointllm_repo)
    add_repo(os.path.join(args.project_root, "EAGLE_EYE"))

    from eagle_eye.ge_data.get_data_all_pointllm import (
        _build_text_query,
        _morton_codes,
        _score_point_tokens,
        _spatially_diverse_topk,
        build_dataset,
        load_checkpoint_tensors_by_prefix,
    )
    from pointllm.model import PointLLMLlamaForCausalLM
    from pointllm.utils import disable_torch_init
    import pointllm.model.pointllm as pointllm_modeling

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    config = AutoConfig.from_pretrained(args.base_model)
    config.point_backbone_config_name = "PointTransformer_8192point_2layer"

    original_loader = pointllm_modeling.cfg_from_yaml_file

    def single_projection_loader(*loader_args, **loader_kwargs):
        point_config = original_loader(*loader_args, **loader_kwargs)
        point_config.model.projection_hidden_layer = 0
        if "projection_hidden_dim" in point_config.model:
            del point_config.model.projection_hidden_dim
        return point_config

    pointllm_modeling.cfg_from_yaml_file = single_projection_loader
    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16,
    )
    load_checkpoint_tensors_by_prefix(
        model, args.base_model, prefixes=("model.point_proj.",)
    )
    model = model.cuda().eval()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.get_model().last_point_group_centers = None
    model.get_model().point_backbone.group_divider.register_forward_hook(
        lambda module, inputs, output: setattr(
            model.get_model(), "last_point_group_centers", output[1].detach()
        )
    )

    class DatasetArgs:
        pass

    dataset_args = DatasetArgs()
    dataset_args.data_path = args.data_path
    dataset_args.anno_path = args.anno_path
    dataset_args.pointnum = 8192
    dataset_args.conversation_types = (
        "single_round,multi_round,detailed_description"
    )
    dataset_args.conv_mode = "vicuna_v1_1"
    dataset_args.start = 0
    dataset_args.end = -1
    dataset, _ = build_dataset(
        dataset_args, tokenizer, model.get_model().point_backbone_config
    )
    sample = dataset[args.sample_index]
    input_ids = sample["input_ids"].unsqueeze(0).cuda()
    point_cloud = sample["point_clouds"].cuda().half()
    labels = sample["labels"]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            point_clouds=point_cloud.unsqueeze(0),
            output_hidden_states=True,
        )

    centers = model.get_model().last_point_group_centers[0].detach().cpu().float()
    point_patch_token = model.get_model().point_backbone_config["point_patch_token"]
    point_mask = input_ids[0].cpu() == point_patch_token
    point_indices = point_mask.nonzero(as_tuple=False).flatten()
    point_count = int(point_indices.numel())
    embeds = outputs.hidden_states[0][0].detach().cpu()
    loss_mask = (labels != -100).long()
    query = _build_text_query(embeds, point_mask, loss_mask=loss_mask, query_mode="all")
    scores = _score_point_tokens(
        embeds[point_indices], query, text_weight=args.text_weight
    )

    keep_count = max(1, min(point_count, int(np.ceil(point_count * args.keep_ratio))))
    selected_group = _spatially_diverse_topk(
        scores[1:], centers, keep_count - 1, args.octree_depth
    )
    selected_mask = torch.zeros(centers.shape[0], dtype=torch.bool)
    selected_mask[selected_group] = True

    remaining = (~selected_mask).nonzero(as_tuple=False).flatten()
    remaining_codes = _morton_codes(centers[remaining], args.octree_depth)
    order = remaining[torch.argsort(remaining_codes)]
    summary_groups = [
        group for group in torch.tensor_split(order, args.summary_count) if group.numel()
    ]
    summary_centers = torch.stack([centers[group].mean(dim=0) for group in summary_groups])

    raw = point_cloud.detach().cpu().float()
    xyz = raw[:, :3]
    rgb = raw[:, 3:6].clamp(0, 1).numpy()
    labels_by_center = nearest_center_labels(xyz, centers)
    retained_points = selected_mask[labels_by_center]
    summary_id = torch.full((centers.shape[0],), -1, dtype=torch.long)
    for index, group in enumerate(summary_groups):
        summary_id[group] = index
    raw_summary_id = summary_id[labels_by_center]

    xyz_np = xyz.numpy()
    retained_np = retained_points.numpy()
    muted = np.full_like(rgb, 0.72)
    overlay = muted.copy()
    overlay[retained_np] = np.array([0.90, 0.12, 0.10])
    cmap = plt.get_cmap("tab10")
    summary_colors = muted.copy()
    for index in range(len(summary_groups)):
        mask = (raw_summary_id.numpy() == index) & (~retained_np)
        summary_colors[mask] = np.array(cmap(index % 10)[:3]) * 0.75 + 0.25

    fig = plt.figure(figsize=(15, 12), facecolor="white")
    panels = [
        ("Original point cloud (8,192 points)", rgb),
        (
            f"Retained token regions ({selected_group.numel()} / {centers.shape[0]} groups)",
            overlay,
        ),
        (
            f"Eight semantic-octree summary regions",
            summary_colors,
        ),
    ]
    for panel_index, (title, colors) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, panel_index, projection="3d")
        scatter(ax, xyz_np, colors)
        if panel_index == 2:
            selected_xyz = centers[selected_group].numpy()
            ax.scatter(
                selected_xyz[:, 0],
                selected_xyz[:, 1],
                selected_xyz[:, 2],
                c="#7a0000",
                s=20,
                marker="o",
                edgecolors="white",
                linewidths=0.35,
                depthshade=False,
            )
        if panel_index == 3:
            summary_xyz = summary_centers.numpy()
            ax.scatter(
                summary_xyz[:, 0],
                summary_xyz[:, 1],
                summary_xyz[:, 2],
                c=[cmap(i % 10) for i in range(len(summary_xyz))],
                s=55,
                marker="X",
                edgecolors="black",
                linewidths=0.5,
                depthshade=False,
            )
        ax.set_title(title, fontsize=13, pad=8)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    scatter(ax, xyz_np, overlay, size=1.6, alpha=0.65)
    summary_xyz = summary_centers.numpy()
    ax.scatter(
        summary_xyz[:, 0],
        summary_xyz[:, 1],
        summary_xyz[:, 2],
        c="#1769aa",
        s=65,
        marker="X",
        edgecolors="white",
        linewidths=0.6,
        depthshade=False,
    )
    ax.set_title(
        f"Compressed representation: {keep_count} retained + "
        f"{len(summary_groups)} summaries",
        fontsize=13,
        pad=8,
    )
    fig.suptitle(
        "Semantic Octree Point-Token Compression",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Red = retained token receptive fields; colored regions / blue X = pooled summaries",
        ha="center",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    os.makedirs(args.output_dir, exist_ok=True)
    image_path = os.path.join(
        args.output_dir, f"semantic_octree_sample_{args.sample_index}.png"
    )
    fig.savefig(image_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        os.path.join(args.output_dir, f"semantic_octree_sample_{args.sample_index}.npz"),
        xyz=xyz_np,
        rgb=rgb,
        retained_point_mask=retained_np,
        group_centers=centers.numpy(),
        selected_group_indices=selected_group.numpy(),
        summary_centers=summary_centers.numpy(),
        raw_summary_id=raw_summary_id.numpy(),
        token_scores=scores[1:].numpy(),
    )
    metadata = {
        "sample_index": args.sample_index,
        "raw_points": int(xyz.shape[0]),
        "point_tokens_before": point_count,
        "spatial_groups": int(centers.shape[0]),
        "selected_point_tokens": keep_count,
        "selected_spatial_groups": int(selected_group.numel()),
        "summary_tokens": len(summary_groups),
        "compressed_point_tokens": keep_count + len(summary_groups),
        "keep_ratio": args.keep_ratio,
        "octree_depth": args.octree_depth,
        "image": image_path,
    }
    with open(
        os.path.join(args.output_dir, f"semantic_octree_sample_{args.sample_index}.json"),
        "w",
        encoding="utf-8",
    ) as dst:
        json.dump(metadata, dst, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
