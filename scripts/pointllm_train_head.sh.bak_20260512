#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_EYE_ROOT="${REPO_ROOT}/EAGLE_EYE"

if [[ -d /root/autodl-tmp ]]; then
  DEFAULT_POINTLLM_REPO="/root/autodl-tmp/pointLLM"
  DEFAULT_BASE_MODEL="/root/autodl-tmp/point7B_v1.1"
  DEFAULT_DATA_DIR="/root/autodl-tmp/pointllm_eagle_data"
  DEFAULT_HEAD_DIR="/root/autodl-tmp/pointllm_eagle_head"
else
  DEFAULT_POINTLLM_REPO="/c/Users/lrz/PointLLM"
  DEFAULT_BASE_MODEL="/f/download/point7B"
  DEFAULT_DATA_DIR="/f/download/pointllm_eagle_data"
  DEFAULT_HEAD_DIR="/f/download/pointllm_eagle_head"
fi

POINTLLM_REPO="${POINTLLM_REPO:-${DEFAULT_POINTLLM_REPO}}"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
DATA_DIR="${DATA_DIR:-${DEFAULT_DATA_DIR}}"
HEAD_DIR="${HEAD_DIR:-${DEFAULT_HEAD_DIR}}"

cd "${EAGLE_EYE_ROOT}"
export PYTHONPATH="${POINTLLM_REPO}:${EAGLE_EYE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m eagle_eye.train.train_pointllm \
  --basepath "${BASE_MODEL}" \
  --pointllm-repo-path "${POINTLLM_REPO}" \
  --tmpdir "${DATA_DIR}" \
  --cpdir "${HEAD_DIR}" \
  --bs "${BATCH_SIZE:-24}" \
  --num-epochs "${NUM_EPOCHS:-20}" \
  --mixed-precision "${MIXED_PRECISION:-no}"
