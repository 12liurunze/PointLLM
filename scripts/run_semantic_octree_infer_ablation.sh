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
RUN_ID="${RUN_ID:-semantic_octree_infer_ablation_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ASSET_ROOT}/results/semantic_octree_ablation/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ASSET_ROOT}/logs/pointllm_eagle_logs_${RUN_ID}}"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"
MASTER_LOG="${RESULT_ROOT}/master.log"

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
export POINT_TOKEN_TEXT_WEIGHT="${POINT_TOKEN_TEXT_WEIGHT:-0.5}"
export POINT_TOKEN_SPATIAL_MODE="${POINT_TOKEN_SPATIAL_MODE:-semantic_octree}"
export POINT_TOKEN_OCTREE_DEPTH="${POINT_TOKEN_OCTREE_DEPTH:-4}"
export POINT_TOKEN_SEMANTIC_TEMPERATURE="${POINT_TOKEN_SEMANTIC_TEMPERATURE:-0.25}"
export DISABLE_POINT_SPATIAL_COMPRESSION=1
unset POINT_SPATIAL_KEEP_RATIO POINT_SPATIAL_NUM_POINTS POINT_SPATIAL_MIN_POINTS

LABELS=(geo sem summary geo_sem geo_summary sem_summary full)
COMPONENTS=(geometry semantic summary geometry,semantic geometry,summary semantic,summary geometry,semantic,summary)

echo "[infer-ablation] run_id=${RUN_ID}" | tee -a "${MASTER_LOG}"
echo "[infer-ablation] result_root=${RESULT_ROOT}" | tee -a "${MASTER_LOG}"
echo "[infer-ablation] head=${HEAD_DIR}" | tee -a "${MASTER_LOG}"

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  comps="${COMPONENTS[$i]}"
  export POINT_TOKEN_COMPONENTS="${comps}"
  export OUTPUT_JSONL="${RESULT_ROOT}/${label}/compare.jsonl"
  export SUMMARY_JSON="${RESULT_ROOT}/${label}/summary.json"
  mkdir -p "${RESULT_ROOT}/${label}"
  echo "[infer-ablation] ===== ${label} components=${comps} start $(date) =====" | tee -a "${MASTER_LOG}"
  bash scripts/pointllm_compare_eagle.sh 2>&1 | tee "${LOG_DIR}/${label}.log"
  cp "${LOG_DIR}/${label}.log" "${RESULT_ROOT}/${label}/compare.log"
  echo "[infer-ablation] ===== ${label} done $(date) summary=${SUMMARY_JSON} =====" | tee -a "${MASTER_LOG}"
  "${PYTHON_BIN}" - <<PY
import json, pathlib
root = pathlib.Path("${RESULT_ROOT}")
map_components = {
    "geo": "geometry",
    "sem": "semantic",
    "summary": "summary",
    "geo_sem": "geometry,semantic",
    "geo_summary": "geometry,summary",
    "sem_summary": "semantic,summary",
    "full": "geometry,semantic,summary",
}
items = []
for label, comps in map_components.items():
    p = root / label / "summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        d["label"] = label
        d["components"] = comps
        items.append(d)
(root / "all_summaries.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
PY
done

echo "[infer-ablation] all done $(date) aggregate=${RESULT_ROOT}/all_summaries.json" | tee -a "${MASTER_LOG}"
