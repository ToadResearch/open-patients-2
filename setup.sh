#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH for this shell session (do not source ~/.bashrc; it may assume an interactive shell).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found on PATH after install."
  echo "[setup] Try: export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\""
  exit 1
fi

echo "[setup] Syncing dependencies (including vllm extra)..."
uv sync --extra vllm

echo "[setup] Installing system build dependencies..."
APT=""
if [ "$(id -u)" -ne 0 ]; then
  APT="sudo"
fi
$APT apt-get update
$APT apt-get install -y build-essential python3-dev

echo "[setup] Installing CUDA compatibility package..."
$APT apt-get install -y cuda-compat-12-9
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "[setup] Note: add /usr/local/cuda-12.9/compat to LD_LIBRARY_PATH for future shells if needed."

echo "[setup] GPU check..."
nvidia-smi
