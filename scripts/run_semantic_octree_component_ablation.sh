#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-pointllm}"
fi

ROOT="${ROOT:-${REPO_ROOT}}"
cd "${ROOT}"

ABLATION_ID="${ABLATION_ID:-semantic_octree_components_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/semantic_octree_ablation/${ABLATION_ID}}"
mkdir -p "${RESULT_ROOT}"
MASTER_LOG="${RESULT_ROOT}/master.log"
AGG_JSON="${RESULT_ROOT}/all_summaries.json"

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
unset POINT_TOKEN_KEEP_RATIOS
export DISABLE_POINT_SPATIAL_COMPRESSION=1
unset POINT_SPATIAL_KEEP_RATIO POINT_SPATIAL_NUM_POINTS POINT_SPATIAL_MIN_POINTS

export BATCH_SIZE="${BATCH_SIZE:-32}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-8}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export LR="${LR:-1e-5}"
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-1}"
export ROLLOUT_WEIGHT="${ROLLOUT_WEIGHT:-0.0}"

export START="${START:-0}"
export END="${END:-10000}"
export EVAL_START="${EVAL_START:-0}"
export EVAL_END="${EVAL_END:--1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-0.0}"
export TOP_K="${TOP_K:-0}"

LABELS=(geo sem summary geo_sem geo_summary sem_summary)
COMPONENTS=(geometry semantic summary geometry,semantic geometry,summary semantic,summary)

printf '[ablation] id=%s\n[ablation] result_root=%s\n' "${ABLATION_ID}" "${RESULT_ROOT}" | tee -a "${MASTER_LOG}"

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  comps="${COMPONENTS[$i]}"
  run_id="${ABLATION_ID}_${label}"
  export POINT_TOKEN_COMPONENTS="${comps}"
  export DATA_DIR="${ROOT}/outputs/pointllm_eagle_data_${run_id}"
  export HEAD_DIR="${ROOT}/outputs/pointllm_eagle_head_${run_id}"
  export LOG_DIR="${ROOT}/logs/pointllm_eagle_logs_${run_id}"
  export OUT_JSONL="${RESULT_ROOT}/${label}/compare.jsonl"
  export SUMMARY_JSON="${RESULT_ROOT}/${label}/summary.json"
  mkdir -p "${RESULT_ROOT}/${label}" "${LOG_DIR}"

  {
    echo "[ablation] ===== ${label} (${comps}) start $(date) ====="
    echo "[ablation] data=${DATA_DIR}"
    echo "[ablation] head=${HEAD_DIR}"
    rm -rf "${DATA_DIR}"

    export OUTPUT_DIR="${DATA_DIR}"
    export TORCH_DTYPE=float16
    echo "[ablation] generate train data $(date)"
    bash scripts/pointllm_generate_data.sh 2>&1 | tee "${LOG_DIR}/01_generate.log"
    echo "[ablation] generated files: $(find "${DATA_DIR}" -type f \( -name '*.ckpt' -o -name '*.pt' \) | wc -l)"

    echo "[ablation] train $(date)"
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
      2>&1 | tee "${LOG_DIR}/02_train.log"

    echo "[ablation] compare fp32 $(date)"
    cd "${ROOT}"
    export TORCH_DTYPE=float32
    export OUTPUT_JSONL="${OUT_JSONL}"
    export SUMMARY_JSON="${SUMMARY_JSON}"
    bash scripts/pointllm_compare_eagle.sh 2>&1 | tee "${LOG_DIR}/03_compare_fp32.log"

    echo "[ablation] cleanup data ${DATA_DIR}"
    rm -rf "${DATA_DIR}"
    echo "[ablation] ===== ${label} done $(date) summary=${SUMMARY_JSON} ====="
  } 2>&1 | tee -a "${MASTER_LOG}"

  "${PYTHON_BIN}" - <<PY
import json, pathlib
root = pathlib.Path("${RESULT_ROOT}")
items = []
for label in ${LABELS[@]@Q}.split():
    p = root / label / "summary.json"
    if p.exists():
        data = json.loads(p.read_text())
        data["label"] = label
        data["components"] = {"geo":"geometry","sem":"semantic","summary":"summary","geo_sem":"geometry,semantic","geo_summary":"geometry,summary","sem_summary":"semantic,summary"}[label]
        items.append(data)
(root / "all_summaries.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
PY

done

echo "[ablation] all done $(date) aggregate=${AGG_JSON}" | tee -a "${MASTER_LOG}"
