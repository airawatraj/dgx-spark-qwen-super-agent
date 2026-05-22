#!/usr/bin/env bash
set -euo pipefail

echo "=== DGX Spark Qwen Setup Check ==="

echo "[1/3] Checking Docker..."
docker version --format 'Docker {{.Server.Version}}' >/dev/null
docker version --format '  Server {{.Server.Version}}'

echo "[2/3] Checking uv / uvx..."
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
uv --version
uvx --version

echo "[3/3] Checking Hugging Face auth..."
uvx hf auth whoami

echo
echo "Setup check complete."
echo "Next: bash setup/download_model.sh"
