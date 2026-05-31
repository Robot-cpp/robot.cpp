#!/usr/bin/env bash
set -euo pipefail

# Calibrate SO101 follower arm. Run robot and teleop calibration separately.
# Usage: ./scripts/calibrate_follower.sh

source "$(dirname "$0")/lib/so101_env.sh"

echo "[calibrate] follower port=${ROBOT_PORT}"

run_lerobot lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}"
