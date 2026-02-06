#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

source ~/.bashrc

# Ensure uv is on PATH for this shell session.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found on PATH after install."
  echo "[setup] Try: export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\""
  exit 1
fi

echo "[setup] Syncing dependencies (including vllm extra)..."
uv sync --extra vllm

echo "[setup] Installing system build dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential python3.12-dev

echo "[setup] Installing CUDA compatibility package..."
sudo apt-get install -y cuda-compat-12-9
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "[setup] GPU check..."
nvidia-smi
