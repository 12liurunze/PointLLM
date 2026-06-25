#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_EYE_ROOT="${REPO_ROOT}/EAGLE_EYE"

DEFAULT_POINTLLM_REPO="/lnt/workspace/jiaming.fjm/lrz/pointLLM"
DEFAULT_BASE_MODEL="/lnt/workspace/jiaming.fjm/lrz/point7B_v1.1"
DEFAULT_POINT_CLOUD_DATA="/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/objaverse_data/8192_npy"
DEFAULT_ANNO_DIR="/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/anno_data"
DEFAULT_ANNOTATION="${DEFAULT_ANNO_DIR}/PointLLM_complex_instruction_70K.json"
DEFAULT_OUTPUT_DIR="/lnt/workspace/jiaming.fjm/lrz/pointllm_eagle_data"

POINTLLM_REPO="${POINTLLM_REPO:-${DEFAULT_POINTLLM_REPO}}"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${DEFAULT_POINT_CLOUD_DATA}}"
ANNO_DIR="${ANNO_DIR:-${DEFAULT_ANNO_DIR}}"
ANNOTATION="${ANNOTATION:-${ANNO_DIR}/PointLLM_complex_instruction_70K.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

# Standalone data generation can run on 4 A100 cards by sharding the dataset.
# Set NUM_GPUS=1 to force the old single-GPU behavior.
NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
POINTLLM_GENERATION_SHARD="${POINTLLM_GENERATION_SHARD:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${POINTLLM_GENERATION_SHARD}" != "1" && "${NUM_GPUS}" -gt 1 ]]; then
  START_INDEX="${START:-0}"
  END_INDEX="${END:--1}"

  if (( END_INDEX <= START_INDEX )); then
    END_INDEX=$("${PYTHON_BIN}" - "${ANNOTATION}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open('r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
PY
)
  fi

  if (( END_INDEX <= START_INDEX )); then
    echo "[generate] WARN: cannot infer a valid END index; falling back to single-GPU generation." >&2
  else
    mkdir -p "${OUTPUT_DIR}/logs"
    total=$((END_INDEX - START_INDEX))
    chunk=$(((total + NUM_GPUS - 1) / NUM_GPUS))
    IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    pids=()

    for ((rank=0; rank<NUM_GPUS; rank++)); do
      shard_start=$((START_INDEX + rank * chunk))
      shard_end=$((shard_start + chunk))
      if (( shard_start >= END_INDEX )); then
        continue
      fi
      if (( shard_end > END_INDEX )); then
        shard_end="${END_INDEX}"
      fi
      gpu_index="${gpu_ids[$rank]:-${rank}}"
      (
        export POINTLLM_GENERATION_SHARD=1
        export NUM_GPUS=1
        export CUDA_VISIBLE_DEVICES="${gpu_index}"
        export INDEX="${rank}"
        export START="${shard_start}"
        export END="${shard_end}"
        bash "${BASH_SOURCE[0]}" > "${OUTPUT_DIR}/logs/generate_rank${rank}.log" 2>&1
      ) &
      pids+=("$!")
      echo "[generate] shard rank=${rank} gpu=${gpu_index} range=[${shard_start},${shard_end}) out=${OUTPUT_DIR}/${rank}"
    done

    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if (( failed != 0 )); then
      echo "[generate] data generation failed; see ${OUTPUT_DIR}/logs/generate_rank*.log" >&2
      exit 1
    fi
    echo "[generate] data generation finished: ${OUTPUT_DIR}"
    exit 0
  fi
fi

cd "${EAGLE_EYE_ROOT}"
export PYTHONPATH="${POINTLLM_REPO}:${EAGLE_EYE_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ "${FORCE_SINGLE_POINT_PROJ:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-single-point-proj)
fi

"${PYTHON_BIN}" -m eagle_eye.ge_data.get_data_all_pointllm \
  --base-model-path "${BASE_MODEL}" \
  --pointllm-repo-path "${POINTLLM_REPO}" \
  --data-path "${POINT_CLOUD_DATA}" \
  --anno-path "${ANNOTATION}" \
  --outdir "${OUTPUT_DIR}" \
  --index "${INDEX:-0}" \
  --start "${START:-0}" \
  --end "${END:--1}" \
  --conversation-types "${CONVERSATION_TYPES:-single_round,multi_round,detailed_description}" \
  --point-backbone-config-name "${POINT_BACKBONE_CONFIG_NAME:-PointTransformer_8192point_2layer}" \
  "${EXTRA_ARGS[@]}" \
  --torch-dtype "${TORCH_DTYPE:-float16}" \
  --point-token-keep-ratio "${POINT_TOKEN_KEEP_RATIO:-0.5}" \
  --point-token-keep-ratios "${POINT_TOKEN_KEEP_RATIOS:-}" \
  --point-token-summary-count "${POINT_TOKEN_SUMMARY_COUNT:-8}" \
  --point-token-text-weight "${POINT_TOKEN_TEXT_WEIGHT:-0.5}" \
  --point-token-spatial-mode "${POINT_TOKEN_SPATIAL_MODE:-none}" \
  --point-token-octree-depth "${POINT_TOKEN_OCTREE_DEPTH:-4}" \
  --point-token-hierarchy-levels "${POINT_TOKEN_HIERARCHY_LEVELS:-1,2,4}" \
  --point-token-density-weight "${POINT_TOKEN_DENSITY_WEIGHT:-0.15}" \
  --point-token-semantic-temperature "${POINT_TOKEN_SEMANTIC_TEMPERATURE:-0.25}" \
  --point-token-components "${POINT_TOKEN_COMPONENTS:-geometry,semantic,summary}" \
  --point-spatial-keep-ratio "${POINT_SPATIAL_KEEP_RATIO:-1.0}" \
  --point-spatial-num-points "${POINT_SPATIAL_NUM_POINTS:-0}" \
  --point-spatial-min-points "${POINT_SPATIAL_MIN_POINTS:-0}" \
  ${DISABLE_POINT_TOKEN_COMPRESSION:+--disable-point-token-compression} \
  ${DISABLE_POINT_SPATIAL_COMPRESSION:+--disable-point-spatial-compression}
