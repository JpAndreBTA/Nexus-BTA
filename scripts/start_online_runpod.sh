#!/usr/bin/env bash
set -euo pipefail

cd /workspace/NexusBTA

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NEXUS_BACKEND_HOST="${NEXUS_BACKEND_HOST:-0.0.0.0}"
export NEXUS_BACKEND_PORT="${NEXUS_BACKEND_PORT:-7861}"
export NEXUS_MODELS_DIR="${NEXUS_MODELS_DIR:-/workspace/NexusBTA/models}"
export NEXUS_COMFY_ROOT="${NEXUS_COMFY_ROOT:-/workspace/NexusBTA/runtime/ComfyUI}"
export NEXUS_COMFY_PYTHON="${NEXUS_COMFY_PYTHON:-$(command -v python)}"
export NEXUS_CUSTOM_NODES_DIR="${NEXUS_CUSTOM_NODES_DIR:-/workspace/NexusBTA/custom_nodes}"
export NEXUS_INPUT_DIR="${NEXUS_INPUT_DIR:-/workspace/NexusBTA/input}"
export NEXUS_OUTPUT_DIR="${NEXUS_OUTPUT_DIR:-/workspace/NexusBTA/output}"
export NEXUS_TEMP_DIR="${NEXUS_TEMP_DIR:-/workspace/NexusBTA/temp}"
export NEXUS_USER_DIR="${NEXUS_USER_DIR:-/workspace/NexusBTA/user}"
export NEXUS_ALLOW_MODEL_DOWNLOADS="${NEXUS_ALLOW_MODEL_DOWNLOADS:-0}"

mkdir -p \
  "$NEXUS_MODELS_DIR" \
  "$NEXUS_CUSTOM_NODES_DIR" \
  "$NEXUS_INPUT_DIR" \
  "$NEXUS_OUTPUT_DIR" \
  "$NEXUS_TEMP_DIR" \
  "$NEXUS_USER_DIR" \
  logs

if [ -d "$NEXUS_CUSTOM_NODES_DIR" ] && [ -d "$NEXUS_COMFY_ROOT/custom_nodes" ]; then
  while IFS= read -r -d '' node_dir; do
    ln -sfn "$node_dir" "$NEXUS_COMFY_ROOT/custom_nodes/$(basename "$node_dir")"
  done < <(find "$NEXUS_CUSTOM_NODES_DIR" -mindepth 1 -maxdepth 1 -type d -print0)
fi

echo "[NEXUS BTA] Starting backend on 0.0.0.0:${NEXUS_BACKEND_PORT}"
python backend/run_backend.py > logs/backend.log 2>&1 &
backend_pid=$!

cleanup() {
  kill "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${NEXUS_BACKEND_PORT}/api/health" >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${NEXUS_BACKEND_PORT}/api/health" >/dev/null; then
  echo "[NEXUS BTA] Backend did not become ready. Last backend log lines:"
  tail -n 80 logs/backend.log || true
  exit 1
fi

curl -fsS -X POST "http://127.0.0.1:${NEXUS_BACKEND_PORT}/api/comfy/start?wait=false" >/dev/null || true

echo "[NEXUS BTA] Local URL: http://127.0.0.1:${NEXUS_BACKEND_PORT}/ui"
echo "[NEXUS BTA] External tunnel disabled. On RunPod, expose HTTP port ${NEXUS_BACKEND_PORT} or direct TCP from the Pod template."
wait "$backend_pid"
