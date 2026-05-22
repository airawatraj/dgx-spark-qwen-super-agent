#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-RedHatAI/Qwen3.6-35B-A3B-NVFP4}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL_ID//\//_}}"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-models--${MODEL_ID//\//--}}"
HF_MODELS_ROOT="${HF_MODELS_ROOT:-$HOME/hf-models}"
MODEL_DIR="${MODEL_DIR:-$HF_MODELS_ROOT/$MODEL_SLUG}"
HUB_MODEL_DIR="${HUB_MODEL_DIR:-$HF_MODELS_ROOT/hub/$MODEL_REPO_DIR}"

echo "=== Downloading $MODEL_ID ==="

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not available."
  echo "Run: bash setup/install.sh"
  exit 1
fi

mkdir -p "$MODEL_DIR"
echo "Downloading to $MODEL_DIR ..."
uvx hf download "$MODEL_ID" --local-dir "$MODEL_DIR"

mkdir -p "$HUB_MODEL_DIR/refs"
printf 'main\n' > "$HUB_MODEL_DIR/refs/main"

echo
echo "Download complete."
echo "Model directory: $MODEL_DIR"
echo "Hub directory:   $HUB_MODEL_DIR"
echo "Next: bash docker/start.sh"
