#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-RedHatAI/Qwen3.6-35B-A3B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}" 
MODEL_SLUG="${MODEL_SLUG:-${MODEL_ID//\//_}}"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-models--${MODEL_ID//\//--}}"
HF_MODELS_ROOT="${HF_MODELS_ROOT:-$HOME/hf-models}"
MODEL_DIR="${MODEL_DIR:-$HF_MODELS_ROOT/$MODEL_SLUG}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_MODELS_ROOT/hub}"
CONTAINER_NAME="${CONTAINER_NAME:-atlas-qwen36}"
ATLAS_IMAGE="${ATLAS_IMAGE:-avarok/atlas-gb10:latest}"
ATLAS_PORT="${ATLAS_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
DROP_CACHES_BEFORE_START="${DROP_CACHES_BEFORE_START:-0}"

echo "=== Atlas Qwen preflight ==="
echo "  Model ID:        $MODEL_ID"
echo "  Model dir:       $MODEL_DIR"
echo "  HF hub cache:    $HF_HUB_CACHE"
echo "  Container:       $CONTAINER_NAME"
echo "  Image:           $ATLAS_IMAGE"
echo "  Port:            $ATLAS_PORT"
echo "  Drop caches:     $DROP_CACHES_BEFORE_START"
echo

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: Model directory not found: $MODEL_DIR"
  echo "Run: bash setup/download_model.sh"
  exit 1
fi

if [[ ! -f "$HF_HUB_CACHE/$MODEL_REPO_DIR/refs/main" ]]; then
  echo "ERROR: Atlas hub ref is missing."
  echo "Run: bash setup/download_model.sh"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Cleaning up existing container..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

if [[ "$DROP_CACHES_BEFORE_START" == "1" ]]; then
  echo "Dropping caches before launch..."
  sudo sync
  sudo sysctl -w vm.drop_caches=3
fi

echo "Starting Atlas..."
docker run -d --name "$CONTAINER_NAME" --gpus all \
  --restart=unless-stopped \
  --shm-size=16gb \
  --ipc=host \
  --network host \
  -v "$HF_HUB_CACHE:/root/.cache/huggingface/hub" \
  -v "$MODEL_DIR:/root/.cache/huggingface/hub/$MODEL_REPO_DIR/snapshots/main" \
  "$ATLAS_IMAGE" \
    serve "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$ATLAS_PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --speculative \
    --num-drafts 1 \
    --enable-prefix-caching \
    --kv-cache-dtype nvfp4

echo
echo "Container started."
echo "Next: docker logs -f $CONTAINER_NAME"
echo "Ready check: curl -sf http://localhost:$ATLAS_PORT/health && echo OK"
