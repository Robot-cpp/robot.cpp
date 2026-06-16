#!/usr/bin/env bash
# Shared SO101 defaults for calibrate / teleoperate / record / client scripts.
# Edit the values below for your machine (serial ports, camera index, dataset hub id, etc.).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_CPP_ROOT="$(cd "${ROOT}/../.." && pwd)"

export CONDA_ENV="${CONDA_ENV:-lerobot-demo}"

export PYTHONPATH="${ROOT}:${ROOT}/lerobot_camera_opencv_crop:${VLA_CPP_ROOT}/robot_client/python:${VLA_CPP_ROOT}/robot_client"

# --- Robot serial ports ---
export ROBOT_PORT="/dev/tty.usbmodem5B3E1195731"
export TELEOP_PORT="/dev/tty.usbmodem5B3E1198201"
export ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
export TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
export ROBOT_USE_DEGREES="${ROBOT_USE_DEGREES:-true}"

# --- Camera ---
export CAMERA_KEY="${CAMERA_KEY:-camera1}"
export MODEL_IMAGE_NAME="${MODEL_IMAGE_NAME:-observation.images.camera1}"
export CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_BACKEND="${CAMERA_BACKEND:-AVFOUNDATION}"
CAMERA_RESIZE_WIDTH="${CAMERA_RESIZE_WIDTH:-224}"
CAMERA_RESIZE_HEIGHT="${CAMERA_RESIZE_HEIGHT:-224}"
CAMERA_WARMUP_S="${CAMERA_WARMUP_S:-5}"

if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  ROBOT_CAMERAS="$(cat <<EOF
{"${CAMERA_KEY}":{"type":"opencv_crop","index_or_path":${CAMERA_INDEX},"width":${CAMERA_WIDTH},"height":${CAMERA_HEIGHT},"fps":${CAMERA_FPS},"backend":"${CAMERA_BACKEND}","resize_width":${CAMERA_RESIZE_WIDTH},"resize_height":${CAMERA_RESIZE_HEIGHT},"warmup_s":${CAMERA_WARMUP_S}}}
EOF
)"
fi
export ROBOT_CAMERAS="${ROBOT_CAMERAS//$'\n'/}"

# --- Inference client (python -m base_policy) ---
export ROBOT_PLATFORM="lerobot_so101"
export SERVER="${SERVER:-127.0.0.1:5555}"
export TASK="${TASK:-grab the block.}"
export FPS="${FPS:-25}"


require_teleop_port() {
  if [[ -z "${TELEOP_PORT}" ]]; then
    echo "[error] TELEOP_PORT is not set (required for teleop/record/leader calibrate)." >&2
    echo "  Edit shell/so101_env.sh or export TELEOP_PORT=/dev/tty.usbmodemYYYY" >&2
    exit 1
  fi
}

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "$@"
  else
    conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
  fi
}

run_lerobot_script() {
  local module="$1"
  shift
  # LeRobot CLI auto-discovers pip-installed lerobot_camera_* plugins only.
  # Import explicitly so dev workflows work with PYTHONPATH (no pip install -e).
  run_python -c '
import runpy
import sys

import lerobot_camera_opencv_crop  # noqa: F401 — registers type "opencv_crop"

module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name="__main__")
' "$module" "$@"
}
