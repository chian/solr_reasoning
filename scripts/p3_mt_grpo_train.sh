#!/usr/bin/env bash
set -euo pipefail


# Simple launcher for P3 + MT-GRPO training that avoids repeated env var setup.
# Usage:
#   1) Copy .p3_env.example to .p3_env and fill in values
#   2) Run: ./scripts/p3_mt_grpo_train.sh
#
# You can override the env file by setting ENV_FILE=/path/to/env

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE=${ENV_FILE:-"$ROOT_DIR/.p3_env"}

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
else
  echo "Env file not found: $ENV_FILE"
  echo "Create it from $ROOT_DIR/.p3_env.example"
  exit 1
fi

: "${TOGETHER_API_KEY:?Set TOGETHER_API_KEY in $ENV_FILE}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in $ENV_FILE}"

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B"}
NUM_GPUS=${NUM_GPUS:-1}
DATASET_JSONL=${DATASET_JSONL:-"$ROOT_DIR/data/demo.jsonl"}

export TOGETHER_API_KEY OPENAI_API_KEY MODEL_NAME NUM_GPUS DATASET_JSONL

exec python "$ROOT_DIR/rl/demo_p3_rl_train.py"


