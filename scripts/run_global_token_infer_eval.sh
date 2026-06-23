#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-/insdata2/jiaming.fjm/lrz}"
ASSET_ROOT="${ASSET_ROOT:-/lnt/workspace/jiaming.fjm/lrz}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-pointllm}"
fi

ROOT="${ROOT:-${CODE_ROOT}/PointLLM}"
cd "${ROOT}"
RUN_ID="${RUN_ID:-global_token_replace_summary_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ASSET_ROOT}/results/global_token/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ASSET_ROOT}/logs/pointllm_eagle_logs_${RUN_ID}}"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export POINTLLM_REPO="${POINTLLM_REPO:-${ASSET_ROOT}/pointLLM}"
export BASE_MODEL="${BASE_MODEL:-${ASSET_ROOT}/point7B_v1.1}"
export POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${ASSET_ROOT}/pointLLM/data/objaverse_data}"
export VAL_JSON="${VAL_JSON:-${ASSET_ROOT}/pointLLM/data/anno_data/PointLLM_brief_description_val_200_GT.json}"
export HEAD_DIR="${HEAD_DIR:-${ASSET_ROOT}/outputs/pointllm_eagle_head_draft5_semantic_octree_5pct_full_bs32}"
export TORCH_DTYPE="${TORCH_DTYPE:-float32}"
export START="${START:-0}"
export END="${END:--1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-0.0}"
export TOP_K="${TOP_K:-0}"

export POINT_TOKEN_KEEP_RATIO="${POINT_TOKEN_KEEP_RATIO:-0.05}"
export POINT_TOKEN_SUMMARY_COUNT="${POINT_TOKEN_SUMMARY_COUNT:-8}"
export POINT_TOKEN_GLOBAL_COUNT="${POINT_TOKEN_GLOBAL_COUNT:-1}"
export POINT_TOKEN_GLOBAL_REPLACE_SUMMARY="${POINT_TOKEN_GLOBAL_REPLACE_SUMMARY:-1}"
export POINT_TOKEN_TEXT_WEIGHT="${POINT_TOKEN_TEXT_WEIGHT:-0.5}"
export POINT_TOKEN_SPATIAL_MODE="${POINT_TOKEN_SPATIAL_MODE:-semantic_octree}"
export POINT_TOKEN_COMPONENTS="${POINT_TOKEN_COMPONENTS:-geometry,semantic,summary}"
export POINT_TOKEN_OCTREE_DEPTH="${POINT_TOKEN_OCTREE_DEPTH:-4}"
export POINT_TOKEN_SEMANTIC_TEMPERATURE="${POINT_TOKEN_SEMANTIC_TEMPERATURE:-0.25}"
export DISABLE_POINT_SPATIAL_COMPRESSION=1
unset POINT_SPATIAL_KEEP_RATIO POINT_SPATIAL_NUM_POINTS POINT_SPATIAL_MIN_POINTS

export OUTPUT_JSONL="${RESULT_ROOT}/compare.jsonl"
export SUMMARY_JSON="${RESULT_ROOT}/summary.json"

echo "[global-token] run_id=${RUN_ID}" | tee "${RESULT_ROOT}/master.log"
echo "[global-token] result_root=${RESULT_ROOT}" | tee -a "${RESULT_ROOT}/master.log"
echo "[global-token] head=${HEAD_DIR}" | tee -a "${RESULT_ROOT}/master.log"
echo "[global-token] config keep=${POINT_TOKEN_KEEP_RATIO} summary=${POINT_TOKEN_SUMMARY_COUNT} global=${POINT_TOKEN_GLOBAL_COUNT} replace=${POINT_TOKEN_GLOBAL_REPLACE_SUMMARY} components=${POINT_TOKEN_COMPONENTS}" | tee -a "${RESULT_ROOT}/master.log"
echo "[global-token] start $(date)" | tee -a "${RESULT_ROOT}/master.log"
bash scripts/pointllm_compare_eagle.sh 2>&1 | tee "${LOG_DIR}/compare.log"
cp "${LOG_DIR}/compare.log" "${RESULT_ROOT}/compare.log"
echo "[global-token] done $(date) summary=${SUMMARY_JSON}" | tee -a "${RESULT_ROOT}/master.log"
