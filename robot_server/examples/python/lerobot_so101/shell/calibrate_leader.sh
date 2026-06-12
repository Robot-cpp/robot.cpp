#!/usr/bin/env bash
set -euo pipefail

# Calibrate SO101 leader (teleop) arm.
# Usage: ./shell/local_calibrate_leader.sh

source "$(dirname "$0")/local_so101_env.sh"

require_teleop_port

echo "[calibrate] leader port=${TELEOP_PORT}"

run_lerobot_script lerobot.scripts.lerobot_calibrate \
  --teleop.type="${TELEOP_TYPE}" \
  --teleop.port="${TELEOP_PORT}"
