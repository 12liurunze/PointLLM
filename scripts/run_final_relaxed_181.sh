#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pointllm
cd /root/autodl-tmp/eagle-eye-eval-point-compress/EAGLE_EYE

export PYTHONPATH=.
export BENCH_LIMIT=181
export BENCH_HEAD=/root/autodl-tmp/pointllm_deterministic_margin_head_v1
export BENCH_TREE=pointllm_wide_45

# Preserve all target point tokens and the full target model. Numerical
# equivalence to serial FP16 greedy decoding is not enforced.
unset POINT_TARGET_KEEP_RATIO
unset POINT_TARGET_SUMMARY_COUNT
unset POINT_TOKEN_KEEP_RATIO
unset POINT_TOKEN_SUMMARY_COUNT
unset POINT_EXACT_REANCHOR
unset POINT_SCALAR_LOGIT_REVERIFY
unset POINT_FP32_ATTENTION

/root/miniconda3/envs/pointllm/bin/python \
  ../scripts/benchmark_lossless_exact.py \
  2>&1 | tee /root/autodl-tmp/final_relaxed_181.log
