#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pointllm
cd /root/autodl-tmp/eagle-eye-eval-point-compress/EAGLE_EYE

export PYTHONPATH=.
export BENCH_LIMIT="${BENCH_LIMIT:-181}"
export BENCH_HEAD=/root/autodl-tmp/pointllm_deterministic_margin_head_v1
export BENCH_TREE=pointllm_wide_45

# Query-conditioned hierarchical octree semantic token compression.
export POINT_TARGET_KEEP_RATIO=0.125
export POINT_TARGET_SUMMARY_COUNT=8
export POINT_TARGET_HIERARCHY_LEVELS=1,2,4
export POINT_TARGET_SEMANTIC_WEIGHT=0.65
export POINT_TARGET_DENSITY_WEIGHT=0.15
export POINT_TARGET_SEMANTIC_TEMPERATURE=0.25

unset POINT_ATTN_IMPLEMENTATION
unset POINT_TARGET_ADAPTIVE_RATIO
unset POINT_TOKEN_KEEP_RATIO
unset POINT_DRAFT_RECOMPUTE_HIDDEN
unset POINT_EXACT_REANCHOR
unset POINT_EXACT_REPLAY
unset POINT_ADAPTIVE_TREE

python ../scripts/benchmark_lossless_exact.py \
  > /root/autodl-tmp/qhostc_fair_keep12p5_eager181.log 2>&1
