#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_EYE_ROOT="${REPO_ROOT}/EAGLE_EYE"

if [[ -d /root/autodl-tmp ]]; then
  DEFAULT_POINTLLM_REPO="/root/autodl-tmp/pointLLM"
  DEFAULT_BASE_MODEL="/root/autodl-tmp/point7B_v1.1"
  DEFAULT_POINT_CLOUD_DATA="/root/autodl-tmp/pointLLM/data/objaverse_data"
  DEFAULT_VAL_JSON="/root/autodl-tmp/pointLLM/data/anno_data/PointLLM_brief_description_val_200_GT.json"
  DEFAULT_HEAD_DIR="/root/autodl-tmp/pointllm_eagle_head"
  DEFAULT_OUTPUT_JSONL="/root/autodl-tmp/pointllm_compare_eagle.jsonl"
  DEFAULT_SUMMARY_JSON="/root/autodl-tmp/pointllm_compare_eagle_summary.json"
else
  DEFAULT_POINTLLM_REPO="/c/Users/lrz/PointLLM"
  DEFAULT_BASE_MODEL="/f/download/point7B"
  DEFAULT_POINT_CLOUD_DATA="/f/download/8192_npy"
  DEFAULT_VAL_JSON="/f/download/PointLLM_brief_description_val_200_GT.json"
  DEFAULT_HEAD_DIR="/f/download/pointllm_eagle_head"
  DEFAULT_OUTPUT_JSONL="/f/download/pointllm_compare_eagle.jsonl"
  DEFAULT_SUMMARY_JSON="/f/download/pointllm_compare_eagle_summary.json"
fi

POINTLLM_REPO="${POINTLLM_REPO:-${DEFAULT_POINTLLM_REPO}}"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${DEFAULT_POINT_CLOUD_DATA}}"
VAL_JSON="${VAL_JSON:-${DEFAULT_VAL_JSON}}"
HEAD_DIR="${HEAD_DIR:-${DEFAULT_HEAD_DIR}}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${DEFAULT_OUTPUT_JSONL}}"
SUMMARY_JSON="${SUMMARY_JSON:-${DEFAULT_SUMMARY_JSON}}"

cd "${EAGLE_EYE_ROOT}"
export PYTHONPATH="${POINTLLM_REPO}:${EAGLE_EYE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS=()
if [[ "${FORCE_SINGLE_POINT_PROJ:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-single-point-proj)
fi

"${PYTHON_BIN}" -m eagle_eye.evaluation.compare_pointllm_eagle \
  --base-model-path "${BASE_MODEL}" \
  --ee-model-path "${HEAD_DIR}" \
  --pointllm-repo-path "${POINTLLM_REPO}" \
  --data-path "${POINT_CLOUD_DATA}" \
  --input-json "${VAL_JSON}" \
  --start "${START:-0}" \
  --end "${END:--1}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --point-backbone-config-name "${POINT_BACKBONE_CONFIG_NAME:-PointTransformer_8192point_2layer}" \
  "${EXTRA_ARGS[@]}" \
  --torch-dtype "${TORCH_DTYPE:-float32}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-0.0}" \
  --top-k "${TOP_K:-0}"
