#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH for this shell session.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found on PATH after install."
  echo "[setup] Try: export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\""
  exit 1
fi

echo "[setup] Syncing dependencies (including vllm extra)..."
uv sync --extra vllm

echo "[setup] GPU check..."
nvidia-smi
