#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vlacpp-lerobot}"

SERVER="${SERVER:-127.0.0.1:5555}"

ROBOT_PORT="${ROBOT_PORT:-/dev/tty.usbmodem5B3E1195731}"
CAMERA_KEY="${CAMERA_KEY:-camera1}"
ROBOT_CAMERAS="${ROBOT_CAMERAS:-{\"camera1\":{\"type\":\"opencv_crop\",\"index_or_path\":0,\"width\":1280,\"height\":720,\"fps\":30,\"backend\":\"AVFOUNDATION\",\"resize_width\":224,\"resize_height\":224,\"center_crop_square_before_resize\":true,\"warmup_s\":5}}}"

TASK="${TASK:-grab the block}"
FPS="${FPS:-25}"
LOOPS="${LOOPS:-0}"

echo "[vlacpp-robot-client] server=${SERVER}"
echo "[vlacpp-robot-client] robot_port=${ROBOT_PORT} camera_key=${CAMERA_KEY} fps=${FPS}"
echo "[vlacpp-robot-client] Start smolvla-server first: bash robot_server/shell/launch_robot_server_mac_cpu.sh"

conda run -n "${CONDA_ENV}" vlacpp-robot-client \
  --server "${SERVER}" \
  --robot-port "${ROBOT_PORT}" \
  --robot-cameras "${ROBOT_CAMERAS}" \
  --camera-key "${CAMERA_KEY}" \
  --task "${TASK}" \
  --fps "${FPS}" \
  --loops "${LOOPS}"
