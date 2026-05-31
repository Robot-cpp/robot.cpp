#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-vlacpp-lerobot}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-10}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.1}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"

echo "[async-server] ${HOST}:${PORT} fps=${FPS}"

conda run -n "${CONDA_ENV}" vlacpp-async-policy-server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}"
