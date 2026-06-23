#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-pointllm}"
fi

RUN_ID="${RUN_ID:-rollout_factor_ablation_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-outputs/pointllm_eagle_data_semantic_octree_5pct_10k}"
OUT_ROOT="${OUT_ROOT:-outputs/rollout_ablation_full_combo/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-logs/rollout_ablation_full_combo/${RUN_ID}}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export POINTLLM_REPO="${POINTLLM_REPO:-${ROOT}/third_party/pointLLM}"
export BASE_MODEL="${BASE_MODEL:-${ROOT}/models/point7B_v1.1}"
export POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${ROOT}/data/objaverse_data}"
export ANNOTATION="${ANNOTATION:-${ROOT}/data/anno_data/PointLLM_complex_instruction_70K.json}"
export VAL_JSON="${VAL_JSON:-${ROOT}/data/anno_data/PointLLM_brief_description_val_200_GT.json}"
export CONVERSATION_TYPES="${CONVERSATION_TYPES:-single_round,multi_round,detailed_description}"

export POINT_TOKEN_KEEP_RATIO="${POINT_TOKEN_KEEP_RATIO:-0.05}"
export POINT_TOKEN_SUMMARY_COUNT="${POINT_TOKEN_SUMMARY_COUNT:-8}"
export POINT_TOKEN_TEXT_WEIGHT="${POINT_TOKEN_TEXT_WEIGHT:-0.5}"
export POINT_TOKEN_SPATIAL_MODE="${POINT_TOKEN_SPATIAL_MODE:-semantic_octree}"
export POINT_TOKEN_OCTREE_DEPTH="${POINT_TOKEN_OCTREE_DEPTH:-4}"
export POINT_TOKEN_SEMANTIC_TEMPERATURE="${POINT_TOKEN_SEMANTIC_TEMPERATURE:-0.25}"
export POINT_TOKEN_COMPONENTS="${POINT_TOKEN_COMPONENTS:-geometry,semantic,summary}"
unset POINT_TOKEN_KEEP_RATIOS
export DISABLE_POINT_SPATIAL_COMPRESSION=1
unset POINT_SPATIAL_KEEP_RATIO POINT_SPATIAL_NUM_POINTS POINT_SPATIAL_MIN_POINTS

export BATCH_SIZE="${BATCH_SIZE:-32}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-8}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export LR="${LR:-1e-5}"
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-3}"
export ROLLOUT_WEIGHT="${ROLLOUT_WEIGHT:-0.18}"
export ROLLOUT_DECAY="${ROLLOUT_DECAY:-0.7}"
export ROLLOUT_ADAPTIVE_MIN="${ROLLOUT_ADAPTIVE_MIN:-0.65}"
export ROLLOUT_ADAPTIVE_MAX="${ROLLOUT_ADAPTIVE_MAX:-1.90}"
export POINT_ROLLOUT_ADAPTIVE=1

export START="${START:-0}"
export END="${END:-10000}"
export EVAL_START="${EVAL_START:-0}"
export EVAL_END="${EVAL_END:--1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-0.0}"
export TOP_K="${TOP_K:-0}"

MASTER_LOG="${OUT_ROOT}/master.log"
AGG_JSON="${OUT_ROOT}/all_summaries.json"

echo "[rollout-ablation] run_id=${RUN_ID}" | tee -a "${MASTER_LOG}"
echo "[rollout-ablation] data_dir=${DATA_DIR}" | tee -a "${MASTER_LOG}"
echo "[rollout-ablation] out_root=${OUT_ROOT}" | tee -a "${MASTER_LOG}"

if [[ ! -d "${DATA_DIR}" ]] || ! find "${DATA_DIR}" -type f \( -name "*.pt" -o -name "*.ckpt" \) | grep -q .; then
  echo "[rollout-ablation] training data not found; generate once $(date)" | tee -a "${MASTER_LOG}"
  export OUTPUT_DIR="${DATA_DIR}"
  bash scripts/pointllm_generate_data.sh 2>&1 | tee "${LOG_ROOT}/00_generate.log"
else
  echo "[rollout-ablation] reuse existing training data ${DATA_DIR}" | tee -a "${MASTER_LOG}"
fi

run_one() {
  local label="$1"
  local length_weight="$2"
  local uncertainty_weight="$3"
  local visual_weight="$4"
  local coverage_weight="$5"

  export HEAD_DIR="${OUT_ROOT}/head_${label}"
  export LOG_DIR="${LOG_ROOT}/${label}"
  export OUT_JSONL="${OUT_ROOT}/${label}/compare.jsonl"
  export SUMMARY_JSON="${OUT_ROOT}/${label}/summary.json"
  export ROLLOUT_LENGTH_WEIGHT="${length_weight}"
  export ROLLOUT_UNCERTAINTY_WEIGHT="${uncertainty_weight}"
  export ROLLOUT_VISUAL_WEIGHT="${visual_weight}"
  export ROLLOUT_COVERAGE_WEIGHT="${coverage_weight}"
  mkdir -p "${LOG_DIR}" "${OUT_ROOT}/${label}"

  {
    echo "[rollout-ablation] ===== ${label} start $(date) ====="
    echo "[rollout-ablation] weights length=${length_weight} uncertainty=${uncertainty_weight} visual=${visual_weight} coverage=${coverage_weight}"
    cd "${ROOT}/EAGLE_EYE"
    export PYTHONPATH="${POINTLLM_REPO}:${ROOT}/EAGLE_EYE:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -m eagle_eye.train.train_pointllm \
      --basepath "${BASE_MODEL}" \
      --pointllm-repo-path "${POINTLLM_REPO}" \
      --tmpdir "${DATA_DIR}" \
      --cpdir "${HEAD_DIR}" \
      --bs "${BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --num-epochs "${NUM_EPOCHS}" \
      --save-freq "${SAVE_FREQ}" \
      --lr "${LR}" \
      --mixed-precision "${MIXED_PRECISION}" \
      --rollout-steps "${ROLLOUT_STEPS}" \
      --rollout-weight "${ROLLOUT_WEIGHT}" \
      --rollout-decay "${ROLLOUT_DECAY}" \
      --rollout-length-weight "${ROLLOUT_LENGTH_WEIGHT}" \
      --rollout-uncertainty-weight "${ROLLOUT_UNCERTAINTY_WEIGHT}" \
      --rollout-visual-weight "${ROLLOUT_VISUAL_WEIGHT}" \
      --rollout-coverage-weight "${ROLLOUT_COVERAGE_WEIGHT}" \
      --rollout-adaptive-min "${ROLLOUT_ADAPTIVE_MIN}" \
      --rollout-adaptive-max "${ROLLOUT_ADAPTIVE_MAX}" \
      --point-rollout-adaptive

    echo "[rollout-ablation] compare fp32 $(date)"
    cd "${ROOT}"
    export TORCH_DTYPE=float32
    export OUTPUT_JSONL="${OUT_JSONL}"
    export SUMMARY_JSON="${SUMMARY_JSON}"
    bash scripts/pointllm_compare_eagle.sh

    if [[ "${KEEP_HEADS:-0}" != "1" ]]; then
      echo "[rollout-ablation] remove head ${HEAD_DIR}"
      rm -rf "${HEAD_DIR}"
    fi
    echo "[rollout-ablation] ===== ${label} done $(date) summary=${SUMMARY_JSON} ====="
  } 2>&1 | tee "${LOG_DIR}/run.log" | tee -a "${MASTER_LOG}" >/dev/null

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
items = []
for path in sorted(root.glob("*/summary.json")):
    data = json.loads(path.read_text())
    data["label"] = path.parent.name
    items.append(data)
(root / "all_summaries.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
PY
}

run_one none 0.0 0.0 0.0 0.0
run_one length 0.30 0.0 0.0 0.0
run_one uncertainty 0.0 0.45 0.0 0.0
run_one visual 0.0 0.0 0.30 0.0
run_one coverage 0.0 0.0 0.0 0.30
run_one length_uncertainty 0.30 0.45 0.0 0.0
run_one length_visual 0.30 0.0 0.30 0.0
run_one length_coverage 0.30 0.0 0.0 0.30
run_one uncertainty_visual 0.0 0.45 0.30 0.0
run_one uncertainty_coverage 0.0 0.45 0.0 0.30
run_one visual_coverage 0.0 0.0 0.30 0.30
run_one length_uncertainty_visual 0.30 0.45 0.30 0.0
run_one length_uncertainty_coverage 0.30 0.45 0.0 0.30
run_one length_visual_coverage 0.30 0.0 0.30 0.30
run_one uncertainty_visual_coverage 0.0 0.45 0.30 0.30
run_one full 0.30 0.45 0.30 0.30

echo "[rollout-ablation] all done $(date) aggregate=${AGG_JSON}" | tee -a "${MASTER_LOG}"
