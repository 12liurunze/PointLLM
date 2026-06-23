#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pointllm
cd /root/autodl-tmp/eagle-eye-eval-point-compress

RUN_ID=20260609_hoss_aggressive
DATA_DIR=/root/autodl-tmp/pointllm_hoss_data_${RUN_ID}
HEAD_DIR=/root/autodl-tmp/pointllm_hoss_head_${RUN_ID}
LOG_DIR=/root/autodl-tmp/pointllm_hoss_logs_${RUN_ID}
mkdir -p "${LOG_DIR}"

export OUTPUT_DIR="${DATA_DIR}"
export START=0
export END=9000
export TORCH_DTYPE=float16
export POINT_TOKEN_KEEP_RATIO=0.125
export POINT_TOKEN_KEEP_RATIOS=0.03125,0.0625,0.0625,0.125,0.125,0.25,0.5
export POINT_TOKEN_SUMMARY_COUNT=8
export POINT_TOKEN_TEXT_WEIGHT=0.5
export POINT_TOKEN_SPATIAL_MODE=hierarchical_octree
export POINT_TOKEN_OCTREE_DEPTH=4
export POINT_TOKEN_HIERARCHY_LEVELS=1,2,4
export POINT_TOKEN_DENSITY_WEIGHT=0.15
export POINT_TOKEN_SEMANTIC_TEMPERATURE=0.25

echo "[HOSS] generate start $(date)"
bash scripts/pointllm_generate_data.sh 2>&1 | tee "${LOG_DIR}/01_generate.log"
echo "[HOSS] generated $(find "${DATA_DIR}" -type f -name '*.ckpt' | wc -l) files"

export DATA_DIR
export HEAD_DIR
export BATCH_SIZE=8
export NUM_EPOCHS=8
export SAVE_FREQ=1
export LEARNING_RATE=1e-5
export MIXED_PRECISION=no
export INIT_HEAD=/root/autodl-tmp/pointllm_eagle_head_compress_fps4096_0p5_8_20260602_203738

echo "[HOSS] train start $(date)"
bash scripts/pointllm_train_head.sh 2>&1 | tee "${LOG_DIR}/02_train.log"

export PYTHONPATH=/root/autodl-tmp/pointLLM:/root/autodl-tmp/eagle-eye-eval-point-compress/EAGLE_EYE
export BENCH_HEAD="${HEAD_DIR}"
export BENCH_LIMIT=181
export BENCH_CONDITIONS=hoss_tree45_keep3p125,hoss_tree45_keep6p25,hoss_tree45_keep12p5,hoss_tree45_keep25,hoss_tree45_keep50

echo "[HOSS] evaluate start $(date)"
python scripts/benchmark_octree_ab.py 2>&1 | tee "${LOG_DIR}/03_evaluate.log"
echo "[HOSS] done $(date)"
