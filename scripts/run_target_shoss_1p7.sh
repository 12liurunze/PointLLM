#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/eagle-eye-eval-point-compress
cd "$ROOT/EAGLE_EYE"

export PYTHONPATH=.
export BENCH_LIMIT="${BENCH_LIMIT:-181}"
export BENCH_CONDITIONS=target_shoss_tree45_keep12p5
export POINT_TARGET_KEEP_RATIO=0.125
export POINT_TARGET_SUMMARY_COUNT=8
export POINT_TARGET_OCTREE_DEPTH=4
unset POINT_PROFILE POINT_ATTN_IMPLEMENTATION

/root/miniconda3/envs/pointllm/bin/python ../scripts/benchmark_octree_ab.py
