#!/bin/bash

set -e

cd "$(dirname "$0")/.."

TARGET_MODEL=${TARGET_MODEL:-RunsenXu/PointLLM_7B_v1.2}
STUDENT_MODEL=${STUDENT_MODEL:-""}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/PointLLM_draft}
DATA_PATH=${DATA_PATH:-./data/objaverse_data}
ANNO_PATH=${ANNO_PATH:-./data/anno_data/PointLLM_complex_instruction_70K.json}

if [ -z "$TARGET_MODEL" ]; then
  echo "[ERROR] TARGET_MODEL is empty. Please set TARGET_MODEL to a valid model path or repo id."
  exit 1
fi

echo "[INFO] TARGET_MODEL=$TARGET_MODEL"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"

EXTRA_ARGS=()
if [ -n "$STUDENT_MODEL" ]; then
  EXTRA_ARGS+=(--student_model_name_or_path "$STUDENT_MODEL")
else
  EXTRA_ARGS+=(--student_num_hidden_layers 12)
fi

PYTHONPATH=$PWD python pointllm/train/train_draft.py \
  --target_model_name_or_path "$TARGET_MODEL" \
  "${EXTRA_ARGS[@]}" \
  --data_path "$DATA_PATH" \
  --anno_path "$ANNO_PATH" \
  --use_color True \
  --conversation_types simple_description single_round multi_round \
  --output_dir "$OUTPUT_DIR" \
  --bf16 True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --max_steps 5000 \
  --logging_steps 10 \
  --save_steps 1000 \
  --kd_alpha 0.7 \
  --kd_temperature 1.0
