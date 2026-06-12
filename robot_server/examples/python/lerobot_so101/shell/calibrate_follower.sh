#!/usr/bin/env bash
set -euo pipefail

# Calibrate SO101 follower arm. Run robot and teleop calibration separately.
# Usage: ./shell/local_calibrate_follower.sh

source "$(dirname "$0")/local_so101_env.sh"

require_robot_port

echo "[calibrate] follower port=${ROBOT_PORT}"

run_lerobot_script lerobot.scripts.lerobot_calibrate \
  --robot.type="${ROBOT_TYPE}" \
  --robot.port="${ROBOT_PORT}"
