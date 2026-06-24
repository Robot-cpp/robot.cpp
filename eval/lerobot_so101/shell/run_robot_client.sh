#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/so101_env.sh"

echo "[robot_sync] platform=${ROBOT_PLATFORM} server=${SERVER}"
echo "[robot_sync] robot_port=${ROBOT_PORT} camera_key=${CAMERA_KEY} fps=${FPS}"
echo "[robot_sync] Start server first: bash robot_server/shell/launch_robot_server_mac_cpu.sh"

run_python "${ROBOT_CPP_ROOT}/eval/lerobot_so101/run_sync.py"
