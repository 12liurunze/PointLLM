#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_EYE_ROOT="${REPO_ROOT}/EAGLE_EYE"

DEFAULT_POINTLLM_REPO="/lnt/workspace/jiaming.fjm/lrz/pointLLM"
DEFAULT_BASE_MODEL="/lnt/workspace/jiaming.fjm/lrz/point7B_v1.1"
DEFAULT_POINT_CLOUD_DATA="/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/objaverse_data/8192_npy"
DEFAULT_ANNO_DIR="/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/anno_data"
DEFAULT_VAL_JSON="${DEFAULT_ANNO_DIR}/PointLLM_brief_description_val_200_GT.json"
DEFAULT_HEAD_DIR="/lnt/workspace/jiaming.fjm/lrz/pointllm_eagle_head"
DEFAULT_OUTPUT_JSONL="/lnt/workspace/jiaming.fjm/lrz/pointllm_compare_eagle.jsonl"
DEFAULT_SUMMARY_JSON="/lnt/workspace/jiaming.fjm/lrz/pointllm_compare_eagle_summary.json"

POINTLLM_REPO="${POINTLLM_REPO:-${DEFAULT_POINTLLM_REPO}}"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${DEFAULT_POINT_CLOUD_DATA}}"
ANNO_DIR="${ANNO_DIR:-${DEFAULT_ANNO_DIR}}"
VAL_JSON="${VAL_JSON:-${ANNO_DIR}/PointLLM_brief_description_val_200_GT.json}"
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
if [[ -n "${TREE_CHOICES:-}" ]]; then
  EXTRA_ARGS+=(--tree-choices "${TREE_CHOICES}")
fi
if [[ -n "${DEVICE_MAP:-}" ]]; then
  EXTRA_ARGS+=(--device-map "${DEVICE_MAP}")
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
