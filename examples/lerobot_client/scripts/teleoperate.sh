#!/usr/bin/env bash
set -euo pipefail

# Teleoperate SO101 follower with leader + front camera (224 crop).
# Usage: ./scripts/teleoperate.sh
# Override: ROBOT_PORT, TELEOP_PORT, CAMERAS_JSON, DISPLAY_DATA=false

source "$(dirname "$0")/lib/so101_env.sh"

DISPLAY_DATA="${DISPLAY_DATA:-true}"

echo "[teleoperate] robot=${ROBOT_PORT} teleop=${TELEOP_PORT} cameras=${CAMERAS_JSON}"

run_lerobot lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --display_data="${DISPLAY_DATA}"
