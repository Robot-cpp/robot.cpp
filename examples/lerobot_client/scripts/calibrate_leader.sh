#!/usr/bin/env bash
set -euo pipefail

# Calibrate SO101 leader (teleop) arm.
# Usage: ./scripts/calibrate_leader.sh

source "$(dirname "$0")/lib/so101_env.sh"

echo "[calibrate] leader port=${TELEOP_PORT}"

run_lerobot lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}"
