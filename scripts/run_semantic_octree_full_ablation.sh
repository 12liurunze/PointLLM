#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CONDA_ENV_NAME="${CONDA_ENV:-pointllm}"
CONDA_SH_PATH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH_PATH}" ]]; then
  source "${CONDA_SH_PATH}"
  conda activate "${CONDA_ENV_NAME}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "[ablation] WARN: conda init script not found; using current Python environment."
fi

RUN_ID="${RUN_ID:-semantic_octree_full_ablation_$(date +%Y%m%d_%H%M%S)}"
WORK_ROOT="${WORK_ROOT:-/lnt/workspace/jiaming.fjm/lrz/semantic_octree_ablation_runs/${RUN_ID}}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/semantic_octree_ablation/${RUN_ID}}"
mkdir -p "${WORK_ROOT}" "${RESULT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export POINTLLM_REPO="${POINTLLM_REPO:-/lnt/workspace/jiaming.fjm/lrz/pointLLM}"
export BASE_MODEL="${BASE_MODEL:-/lnt/workspace/jiaming.fjm/lrz/point7B_v1.1}"
export POINT_CLOUD_DATA="${POINT_CLOUD_DATA:-/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/objaverse_data/8192_npy}"
export ANNO_DIR="${ANNO_DIR:-/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/anno_data}"
export ANNOTATION="${ANNOTATION:-${ANNO_DIR}/PointLLM_complex_instruction_70K.json}"
export VAL_JSON="${VAL_JSON:-${ANNO_DIR}/PointLLM_brief_description_val_200_GT.json}"
export CONVERSATION_TYPES="${CONVERSATION_TYPES:-single_round,multi_round,detailed_description}"

# 4-card A100 defaults. Override NUM_GPUS/CUDA_VISIBLE_DEVICES when needed.
export NUM_GPUS="${NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

GEN_START="${START:-0}"
GEN_END="${END:--1}"
export EVAL_START="${EVAL_START:-0}"
export EVAL_END="${EVAL_END:--1}"
# BATCH_SIZE is per GPU under torchrun. Default 8 x 4 GPUs = global batch 32.
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-8}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export LR="${LR:-1e-5}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"

export POINT_TOKEN_TEXT_WEIGHT="${POINT_TOKEN_TEXT_WEIGHT:-0.5}"
export POINT_TOKEN_SEMANTIC_TEMPERATURE="${POINT_TOKEN_SEMANTIC_TEMPERATURE:-0.25}"
export POINT_TOKEN_SPATIAL_MODE=semantic_octree
export DISABLE_POINT_SPATIAL_COMPRESSION=1
unset POINT_TOKEN_KEEP_RATIOS
unset POINT_SPATIAL_KEEP_RATIO POINT_SPATIAL_NUM_POINTS POINT_SPATIAL_MIN_POINTS

ABLATION_GROUP_LIST="${ABLATION_GROUPS:-baseline,components,ratio}"
KEEP_DATA="${KEEP_DATA:-0}"
KEEP_HEADS="${KEEP_HEADS:-1}"
MASTER_LOG="${RESULT_ROOT}/master.log"
MANIFEST="${RESULT_ROOT}/manifest.jsonl"

group_enabled() {
  [[ ",${ABLATION_GROUP_LIST}," == *,"$1",* ]]
}

append_summary() {
  "${PYTHON_BIN}" - "${RESULT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
items = []
for path in sorted(root.glob("*/summary.json")):
    data = json.loads(path.read_text())
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        data.update(json.loads(metadata_path.read_text()))
    items.append(data)
(root / "all_summaries.json").write_text(
    json.dumps(items, indent=2, ensure_ascii=False)
)
PY
}

run_data_generation() {
  local data_dir="$1"
  local log_dir="$2"

  rm -rf "${data_dir}"
  mkdir -p "${data_dir}" "${log_dir}"

  local start="${GEN_START}"
  local end="${GEN_END}"
  if (( NUM_GPUS > 1 && end > start )); then
    local total=$((end - start))
    local chunk=$(((total + NUM_GPUS - 1) / NUM_GPUS))
    IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    local pids=()

    for ((rank=0; rank<NUM_GPUS; rank++)); do
      local shard_start=$((start + rank * chunk))
      local shard_end=$((shard_start + chunk))
      if (( shard_start >= end )); then
        continue
      fi
      if (( shard_end > end )); then
        shard_end="${end}"
      fi
      local gpu_index="${gpu_ids[$rank]:-${rank}}"
      (
        export CUDA_VISIBLE_DEVICES="${gpu_index}"
        export POINTLLM_GENERATION_SHARD=1
        export OUTPUT_DIR="${data_dir}"
        export INDEX="${rank}"
        export START="${shard_start}"
        export END="${shard_end}"
        export TORCH_DTYPE=float16
        bash scripts/pointllm_generate_data.sh > "${log_dir}/01_generate_rank${rank}.log" 2>&1
      ) &
      pids+=("$!")
      echo "[ablation] generate shard rank=${rank} gpu=${gpu_index} range=[${shard_start},${shard_end})"
    done

    local failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if (( failed != 0 )); then
      echo "[ablation] data generation failed; see ${log_dir}/01_generate_rank*.log" >&2
      return 1
    fi
  else
    export OUTPUT_DIR="${data_dir}"
    export INDEX="0"
    export START="${GEN_START}"
    export END="${GEN_END}"
    export TORCH_DTYPE=float16
    bash scripts/pointllm_generate_data.sh 2>&1 | tee "${log_dir}/01_generate.log"
  fi
}

run_training() {
  local data_dir="$1"
  local head_dir="$2"
  local log_dir="$3"

  cd "${ROOT}/EAGLE_EYE"
  export PYTHONPATH="${POINTLLM_REPO}:${ROOT}/EAGLE_EYE:${PYTHONPATH:-}"
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    -m eagle_eye.train.train_pointllm \
    --basepath "${BASE_MODEL}" \
    --pointllm-repo-path "${POINTLLM_REPO}" \
    --tmpdir "${data_dir}" \
    --cpdir "${head_dir}" \
    --bs "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --num-epochs "${NUM_EPOCHS}" \
    --save-freq "${SAVE_FREQ}" \
    --lr "${LR}" \
    --mixed-precision "${MIXED_PRECISION}" \
    2>&1 | tee "${log_dir}/02_train_4gpu.log"
  cd "${ROOT}"
}

run_variant() {
  local label="$1"
  local components="$2"
  local keep_ratio="$3"
  local summary_count="$4"
  local depth="$5"
  local disable_compression="$6"

  local data_dir="${WORK_ROOT}/data_${label}"
  local head_dir="${WORK_ROOT}/head_${label}"
  local log_dir="${WORK_ROOT}/logs_${label}"
  local result_dir="${RESULT_ROOT}/${label}"
  mkdir -p "${log_dir}" "${result_dir}"

  export POINT_TOKEN_COMPONENTS="${components}"
  export POINT_TOKEN_KEEP_RATIO="${keep_ratio}"
  export POINT_TOKEN_SUMMARY_COUNT="${summary_count}"
  export POINT_TOKEN_OCTREE_DEPTH="${depth}"
  if [[ "${disable_compression}" == "1" ]]; then
    export DISABLE_POINT_TOKEN_COMPRESSION=1
  else
    unset DISABLE_POINT_TOKEN_COMPRESSION
  fi

  cat > "${result_dir}/metadata.json" <<EOF
{
  "label": "${label}",
  "components": "${components}",
  "keep_ratio": ${keep_ratio},
  "summary_count": ${summary_count},
  "octree_depth": ${depth},
  "compression_disabled": ${disable_compression},
  "num_gpus": ${NUM_GPUS},
  "per_gpu_batch_size": ${BATCH_SIZE},
  "mixed_precision": "${MIXED_PRECISION}"
}
EOF
  cat "${result_dir}/metadata.json" >> "${MANIFEST}"
  printf '\n' >> "${MANIFEST}"

  {
    echo "[ablation] ===== ${label} start $(date) ====="
    echo "[ablation] components=${components} keep=${keep_ratio} summary=${summary_count} depth=${depth}"
    echo "[ablation] data=${data_dir}"
    echo "[ablation] head=${head_dir}"
    echo "[ablation] num_gpus=${NUM_GPUS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} batch_size_per_gpu=${BATCH_SIZE}"

    rm -rf "${head_dir}"
    run_data_generation "${data_dir}" "${log_dir}"
    run_training "${data_dir}" "${head_dir}" "${log_dir}"

    export HEAD_DIR="${head_dir}"
    export OUTPUT_JSONL="${result_dir}/compare.jsonl"
    export SUMMARY_JSON="${result_dir}/summary.json"
    export START="${EVAL_START}"
    export END="${EVAL_END}"
    export TORCH_DTYPE=float32
    bash scripts/pointllm_compare_eagle.sh 2>&1 | tee "${log_dir}/03_compare_fp32.log"

    if [[ "${KEEP_DATA}" != "1" ]]; then
      rm -rf "${data_dir}"
    fi
    if [[ "${KEEP_HEADS}" != "1" ]]; then
      rm -rf "${head_dir}"
    fi
    echo "[ablation] ===== ${label} done $(date) ====="
  } 2>&1 | tee -a "${MASTER_LOG}"

  append_summary
}

echo "[ablation] run_id=${RUN_ID}" | tee -a "${MASTER_LOG}"
echo "[ablation] groups=${ABLATION_GROUP_LIST}" | tee -a "${MASTER_LOG}"
echo "[ablation] pointllm_repo=${POINTLLM_REPO}" | tee -a "${MASTER_LOG}"
echo "[ablation] base_model=${BASE_MODEL}" | tee -a "${MASTER_LOG}"
echo "[ablation] point_cloud_data=${POINT_CLOUD_DATA}" | tee -a "${MASTER_LOG}"
echo "[ablation] annotation=${ANNOTATION}" | tee -a "${MASTER_LOG}"
echo "[ablation] num_gpus=${NUM_GPUS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}" | tee -a "${MASTER_LOG}"

if group_enabled baseline; then
  run_variant uncompressed geometry,semantic,summary 1.0 0 4 1
fi

if group_enabled components; then
  run_variant geometry geometry 0.05 8 4 0
  run_variant semantic semantic 0.05 8 4 0
  run_variant summary summary 0.05 8 4 0
  run_variant geometry_semantic geometry,semantic 0.05 8 4 0
  run_variant geometry_summary geometry,summary 0.05 8 4 0
  run_variant semantic_summary semantic,summary 0.05 8 4 0
  run_variant full geometry,semantic,summary 0.05 8 4 0
fi

if group_enabled ratio; then
  run_variant ratio_001 geometry,semantic,summary 0.01 8 4 0
  run_variant ratio_003 geometry,semantic,summary 0.03 8 4 0
fi

if group_enabled summary; then
  run_variant summary_000 geometry,semantic,summary 0.05 0 4 0
  run_variant summary_004 geometry,semantic,summary 0.05 4 4 0
  run_variant summary_016 geometry,semantic,summary 0.05 16 4 0
fi

if group_enabled depth; then
  run_variant depth_2 geometry,semantic,summary 0.05 8 2 0
  run_variant depth_3 geometry,semantic,summary 0.05 8 3 0
  run_variant depth_5 geometry,semantic,summary 0.05 8 5 0
fi

echo "[ablation] all done $(date)" | tee -a "${MASTER_LOG}"
echo "[ablation] aggregate=${RESULT_ROOT}/all_summaries.json" | tee -a "${MASTER_LOG}"
