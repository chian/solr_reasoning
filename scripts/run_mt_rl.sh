#!/usr/bin/env bash
set -euo pipefail

# Minimal wrapper to invoke the adapter in Python.
# Usage example:
#   MODEL_NAME="Qwen/Qwen2.5-7B" NUM_GPUS=2 ./scripts/run_mt_rl.sh

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B"}
NUM_GPUS=${NUM_GPUS:-1}

PYTHON=${PYTHON:-python}

$PYTHON - <<'PY'
from integrations.mt_rl.adapter import run_mt_grpo
import os

model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B")
num_gpus = int(os.environ.get("NUM_GPUS", "1"))

rc = run_mt_grpo(
    model_name=model_name,
    num_gpus=num_gpus,
)

raise SystemExit(rc)
PY


