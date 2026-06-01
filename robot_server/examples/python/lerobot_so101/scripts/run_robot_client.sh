#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/so101_env.sh"

require_robot_port

SERVER="${SERVER:-127.0.0.1:5555}"
TASK="${TASK:-grab the block.}"
FPS="${FPS:-25}"
LOOPS="${LOOPS:-0}"

echo "[lerobot-so101-client] server=${SERVER}"
echo "[lerobot-so101-client] robot_port=${ROBOT_PORT} camera_key=${CAMERA_KEY} cameras=${CAMERAS_JSON} fps=${FPS}"
echo "[lerobot-so101-client] Start server first: bash robot_server/shell/launch_robot_server_mac_cpu.sh"

run_lerobot lerobot-so101-client \
  --server "${SERVER}" \
  --robot-port "${ROBOT_PORT}" \
  --robot-cameras "${ROBOT_CAMERAS}" \
  --camera-key "${CAMERA_KEY}" \
  --task "${TASK}" \
  --fps "${FPS}" \
  --loops "${LOOPS}"
