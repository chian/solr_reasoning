#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATASET_JSONL=path/to/data.jsonl MODEL_NAME="Qwen/Qwen2.5-7B" NUM_GPUS=1 ./scripts/run_mt_grpo_bvbrc.sh

DATASET_JSONL=${DATASET_JSONL:?DATASET_JSONL is required}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B"}
NUM_GPUS=${NUM_GPUS:-1}

LR=${LR:-1e-6}
GENS=${GENS:-8}
BSZ=${BSZ:-2}
ACC=${ACC:-2}
ITERS=${ITERS:-1}
STEPS=${STEPS:-100}
BETA=${BETA:-0.0}
TURN_COEF=${TURN_COEF:-1.0}

ACCEL_CFG=${ACCEL_CFG:-external/Multi-Turn-RL-Agent/configs/zero3.yaml}

accelerate launch --config-file "$ACCEL_CFG" --num-processes "$((NUM_GPUS>1?NUM_GPUS-1:1))" \
  external/Multi-Turn-RL-Agent/verifiers/examples/bvbrc_qa.py \
  --dataset_jsonl "$DATASET_JSONL" \
  --model_name "$MODEL_NAME" \
  --num_gpus "$NUM_GPUS" \
  --learning_rate "$LR" \
  --num_generations "$GENS" \
  --per_device_train_batch_size "$BSZ" \
  --grad_accum_steps "$ACC" \
  --num_iterations "$ITERS" \
  --max_steps "$STEPS" \
  --beta "$BETA" \
  --turn_advantage_coef "$TURN_COEF"


