#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_EYE_ROOT="${REPO_ROOT}/EAGLE_EYE"
ASSET_ROOT="${ASSET_ROOT:-/lnt/workspace/jiaming.fjm/lrz}"

DEFAULT_POINTLLM_REPO="${ASSET_ROOT}/pointLLM"
DEFAULT_BASE_MODEL="${ASSET_ROOT}/point7B_v1.1"
DEFAULT_POINT_CLOUD_DATA="${ASSET_ROOT}/pointLLM/data/objaverse_data"
DEFAULT_ANNOTATION="${ASSET_ROOT}/pointLLM/data/anno_data/PointLLM_complex_instruction_70K.json"
DEFAULT_OUTPUT_DIR="${ASSET_ROOT}/outputs/pointllm_eagle_data"

POINTLLM_REPO="${POINTLLM_REPO:-${DEFAULT_POINTLLM_REPO}}"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-${DEFAULT_POINT_CLOUD_DATA}}"
ANNOTATION="${ANNOTATION:-${DEFAULT_ANNOTATION}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

cd "${EAGLE_EYE_ROOT}"
export PYTHONPATH="${POINTLLM_REPO}:${EAGLE_EYE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
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
  --end "${END:-0}" \
  --conversation-types "${CONVERSATION_TYPES:-single_round,multi_round,detailed_description}" \
  --point-backbone-config-name "${POINT_BACKBONE_CONFIG_NAME:-PointTransformer_8192point_2layer}" \
  "${EXTRA_ARGS[@]}" \
  --torch-dtype "${TORCH_DTYPE:-float32}" \
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
