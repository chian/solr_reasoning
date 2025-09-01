wo d il #!/usr/bin/env bash
set -euo pipefail

# Portable env setup for P3 + MT-GRPO
# - Uses this script's location to find the repo root
# - Works with conda if available, else falls back to venv
# - Installs pinned deps to avoid NumPy/PyArrow ABI issues
# - Optional GPU (CUDA) install for Torch and vLLM on Linux
#
# Usage (defaults):
#   ./scripts/setup_p3_rl_env.sh
#
# Override via env vars:
#   ENV_NAME=p3_rl PYTHON_VERSION=3.11 INSTALL_VLLM=1 CUDA_VERSION=cu121 TORCH_VERSION=2.4.0 ./scripts/setup_p3_rl_env.sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ENV_NAME=${ENV_NAME:-p3_rl}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
INSTALL_VLLM=${INSTALL_VLLM:-0}         # 1 to install vLLM (Linux + NVIDIA)
CUDA_VERSION=${CUDA_VERSION:-cu121}      # cu118|cu121 (only used if INSTALL_VLLM=1)
TORCH_VERSION=${TORCH_VERSION:-2.4.0}

echo "Repo root: $ROOT_DIR"
echo "Env name:  $ENV_NAME"
echo "Python:    $PYTHON_VERSION"
echo "vLLM:      ${INSTALL_VLLM} (CUDA: ${CUDA_VERSION})"

have_conda=0
if command -v conda >/dev/null 2>&1; then
  have_conda=1
fi

if [ "$have_conda" -eq 1 ]; then
  echo "Using conda environment: $ENV_NAME"
  # Ensure conda shell functions are available
  eval "$(conda shell.bash hook)"
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
  conda activate "$ENV_NAME"
else
  echo "conda not found; using venv at $ROOT_DIR/.venv_$ENV_NAME"
  PY_BIN="python${PYTHON_VERSION%.*}"
  if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    PY_BIN=python3
  fi
  "$PY_BIN" -m venv "$ROOT_DIR/.venv_$ENV_NAME"
  # shellcheck disable=SC1090
  . "$ROOT_DIR/.venv_$ENV_NAME/bin/activate"
fi

python -V
pip install -U pip setuptools wheel

# Core pins to avoid ABI issues and ensure trainer compatibility
pip install "numpy==1.26.4" "pyarrow==14.0.2"
pip install "transformers==4.56.0" "accelerate==1.10.1" "datasets==4.0.0" "trl==0.22.1"

# Torch
if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "Installing Torch (CUDA) ${TORCH_VERSION}"
  if [ "$CUDA_VERSION" = "cu121" ]; then
    pip install --index-url https://download.pytorch.org/whl/cu121 "torch==${TORCH_VERSION}" "torchvision==0.19.0" "torchaudio==2.4.0"
  elif [ "$CUDA_VERSION" = "cu118" ]; then
    pip install --index-url https://download.pytorch.org/whl/cu118 "torch==${TORCH_VERSION}" "torchvision==0.19.0" "torchaudio==2.4.0"
  else
    echo "Unknown CUDA_VERSION: $CUDA_VERSION; installing default CPU wheels"
    pip install "torch==${TORCH_VERSION}"
  fi
else
  echo "Installing Torch (CPU) ${TORCH_VERSION}"
  pip install "torch==${TORCH_VERSION}"
fi

# Project tool deps
pip install "httpx>=0.27" mcp openai together

# Optional vLLM/flash-attn (Linux + NVIDIA recommended)
if [ "$INSTALL_VLLM" = "1" ]; then
  if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Installing vLLM + flash-attn"
    pip install "vllm==0.5.4.post1" "flash-attn==2.5.8" --no-build-isolation
  else
    echo "Skipping vLLM: requires Linux + NVIDIA GPU"
  fi
fi

echo
echo "Setup complete. Activate env and launch training:"
if [ "$have_conda" -eq 1 ]; then
  echo "  conda activate $ENV_NAME"
else
  echo "  source $ROOT_DIR/.venv_$ENV_NAME/bin/activate"
fi
echo "  cd $ROOT_DIR"
echo "  cp .p3_env.example .p3_env  # ensure keys and DATASET_JSONL are set"
echo "  ./scripts/p3_mt_grpo_train.sh"


