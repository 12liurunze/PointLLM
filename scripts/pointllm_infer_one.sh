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
  DEFAULT_OUTPUT_JSONL="/root/autodl-tmp/pointllm_ee_answers.jsonl"
else
  DEFAULT_POINTLLM_REPO="/c/Users/lrz/PointLLM"
  DEFAULT_BASE_MODEL="/f/download/point7B"
  DEFAULT_POINT_CLOUD_DATA="/f/download/8192_npy"
  DEFAULT_VAL_JSON="/f/download/PointLLM_brief_description_val_200_GT.json"
  DEFAULT_HEAD_DIR="/f/download/pointllm_eagle_head"
  DEFAULT_OUTPUT_JSONL="/f/download/pointllm_ee_answers.jsonl"
fi

POINTLLM_REPO="${POINTLLM_REPO:-${DEFAULT_POINTLLM_REPO}}"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${DEFAULT_POINT_CLOUD_DATA}}"
VAL_JSON="${VAL_JSON:-${DEFAULT_VAL_JSON}}"
HEAD_DIR="${HEAD_DIR:-${DEFAULT_HEAD_DIR}}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${DEFAULT_OUTPUT_JSONL}}"

VAL_INDEX="${VAL_INDEX:-0}"
if [[ "${AUTO_READ_OBJECT_IDS:-1}" == "1" ]]; then
  INPUT_ARGS=(--input-json "${VAL_JSON}" --start "${START:-${VAL_INDEX}}" --end "${END:-$((VAL_INDEX + 1))}")
elif [[ $# -ge 1 ]]; then
  OBJECT_ID="$1"
  QUESTION="${2:-${QUESTION:-Describe this 3D object in detail.}}"
  INPUT_ARGS=(--object-id "${OBJECT_ID}" --question "${QUESTION}")
else
  read -r OBJECT_ID QUESTION < <("${PYTHON_BIN:-python}" - "${VAL_JSON}" "${VAL_INDEX}" <<'PY'
import json
import re
import sys

path, index = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
sample = data[index]
question = sample.get("conversations", [{}])[0].get("value", "Describe this 3D object in detail.")
question = re.sub(r"\s*<point>\s*", "", question).strip()
print(sample["object_id"], question)
PY
)
  INPUT_ARGS=(--object-id "${OBJECT_ID}" --question "${QUESTION}")
fi

cd "${EAGLE_EYE_ROOT}"
export PYTHONPATH="${POINTLLM_REPO}:${EAGLE_EYE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS=()
if [[ "${FORCE_SINGLE_POINT_PROJ:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-single-point-proj)
fi

"${PYTHON_BIN}" -m eagle_eye.evaluation.gen_ee_answer_pointllm \
  --base-model-path "${BASE_MODEL}" \
  --ee-model-path "${HEAD_DIR}" \
  --pointllm-repo-path "${POINTLLM_REPO}" \
  --data-path "${POINT_CLOUD_DATA}" \
  "${INPUT_ARGS[@]}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --point-backbone-config-name "${POINT_BACKBONE_CONFIG_NAME:-PointTransformer_8192point_2layer}" \
  "${EXTRA_ARGS[@]}" \
  --torch-dtype "${TORCH_DTYPE:-float32}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --max-length "${MAX_LENGTH:-2048}"
