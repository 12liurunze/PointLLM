#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pointllm
cd /root/autodl-tmp/eagle-eye-eval-point-compress

DATA_DIR=/root/autodl-tmp/pointllm_deterministic_full_data_v1
HEAD_DIR=/root/autodl-tmp/pointllm_deterministic_margin_head_v1
LOG_DIR=/root/autodl-tmp/pointllm_deterministic_margin_logs_v1
mkdir -p "${LOG_DIR}"

export OUTPUT_DIR="${DATA_DIR}"
export START=9000
export END=13000
export TORCH_DTYPE=float16
export DISABLE_POINT_TOKEN_COMPRESSION=1
export DISABLE_POINT_SPATIAL_COMPRESSION=1

echo "[deterministic-full] generate start $(date)"
bash scripts/pointllm_generate_data.sh 2>&1 | tee "${LOG_DIR}/01_generate.log"

cd EAGLE_EYE
export PYTHONPATH=/root/autodl-tmp/pointLLM:/root/autodl-tmp/eagle-eye-eval-point-compress/EAGLE_EYE
echo "[deterministic-full] train start $(date)"
python -m eagle_eye.train.train_pointllm \
  --basepath /root/autodl-tmp/point7B_v1.1 \
  --pointllm-repo-path /root/autodl-tmp/pointLLM \
  --tmpdir "${DATA_DIR}" \
  --cpdir "${HEAD_DIR}" \
  --init-head /root/autodl-tmp/pointllm_eagle_head_compress_fps4096_0p5_8_20260602_203738 \
  --bs 4 \
  --num-epochs 2 \
  --save-freq 1 \
  --lr 5e-6 \
  --mixed-precision no \
  --no-data-noise \
  --acceptance-margin-weight 0.2 \
  --acceptance-margin 1.0 \
  2>&1 | tee "${LOG_DIR}/02_train.log"
echo "[deterministic-full] done $(date)"
