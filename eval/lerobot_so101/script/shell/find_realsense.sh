#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/script/shell/so101_env.sh"

echo "[find-realsense] Listing RealSense devices via pyrealsense2..."
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "[find-realsense] macOS: try 'sudo rs-enumerate-devices -s' if this fails."
fi
set +e
FIND_OUTPUT="$(run_lerobot_script lerobot.scripts.lerobot_find_cameras realsense 2>&1)"
FIND_STATUS=$?
set -e
printf '%s\n' "${FIND_OUTPUT}"

if [[ "${FIND_STATUS}" -eq 0 ]] && printf '%s\n' "${FIND_OUTPUT}" | grep -qi "serial"; then
  echo
  echo "Set REALSENSE_SERIAL in script/shell/so101_env.sh, then re-run camera test."
  exit 0
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo
  echo "[find-realsense] pyrealsense2 failed on macOS (common: failed to set power state / No device connected)."
  echo "[find-realsense] macOS needs sudo for USB. Run:"
  echo "  sudo rs-enumerate-devices -s"
  echo
  echo "Use the line 'Serial Number:' (e.g. 141722072266)."
  echo "Do NOT use 'Asic Serial Number' or 'Firmware Update Id'."
  echo
  echo "Then enter a root shell and run the camera test:"
  echo "  sudo -s"
  echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate lerobot-demo"
  echo "  cd \"${ROOT}\""
  echo "  export REALSENSE_SUDO=0 CAMERA_TYPE=realsense REALSENSE_SERIAL=你的序列号"
  echo "  FRAMES=5 ./test/run_camera_test.sh --preview"
  echo
  echo "First-time setup: bash script/shell/setup_macos_realsense.sh"
  exit 0
fi

exit "${FIND_STATUS}"
