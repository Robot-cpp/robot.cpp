#!/usr/bin/env bash
set -euo pipefail

# Teleoperate SO101 follower with leader + front camera (224 crop).
# Usage: ./shell/local_teleoperate.sh
# Override: ROBOT_PORT, TELEOP_PORT, ROBOT_CAMERAS, DISPLAY_DATA=false

source "$(dirname "$0")/local_so101_env.sh"

require_robot_port
require_teleop_port

echo "[teleoperate] robot=${ROBOT_PORT} teleop=${TELEOP_PORT} camera_key=${CAMERA_KEY}"

run_lerobot_script lerobot.scripts.lerobot_teleoperate \
  --robot.type="${ROBOT_TYPE}" \
  --robot.port="${ROBOT_PORT}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type="${TELEOP_TYPE}" \
  --teleop.port="${TELEOP_PORT}" \
  --display_data="${DISPLAY_DATA}"
