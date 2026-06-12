#!/usr/bin/env bash
# Shared SO101 defaults for calibrate / teleoperate / record / client scripts.
# Edit the values below for your machine (serial ports, camera index, dataset hub id, etc.).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_CPP_ROOT="$(cd "${ROOT}/../../../.." && pwd)"

if [[ -f "${VLA_CPP_ROOT}/local_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${VLA_CPP_ROOT}/local_env.sh"
fi

export LEROBOT_SO101_ROOT="${LEROBOT_SO101_ROOT:-${ROOT}}"
export CONDA_ENV="${CONDA_ENV:-mini-client}"

export PYTHONPATH="${LEROBOT_SO101_ROOT}:${LEROBOT_SO101_ROOT}/..:${LEROBOT_SO101_ROOT}/camera:${VLA_CPP_ROOT}/robot_server:${VLA_CPP_ROOT}/robot_server/examples/python"

# --- Robot serial ports ---
export ROBOT_PORT="${ROBOT_PORT:?ROBOT_PORT must be set}"
export TELEOP_PORT="${TELEOP_PORT:?TELEOP_PORT must be set}"
export ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
export TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
export ROBOT_USE_DEGREES="${ROBOT_USE_DEGREES:-true}"

# --- Camera ---
export CAMERA_KEY="${CAMERA_KEY:-camera1}"
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

# --- Inference client ---
export SERVER="${SERVER:-127.0.0.1:5555}"
export TASK="${TASK:-grab the block.}"
export FPS="${FPS:-25}"
export LOOPS="${LOOPS:-0}"

require_robot_port() {
  if [[ -z "${ROBOT_PORT}" ]]; then
    echo "[error] ROBOT_PORT is not set." >&2
    echo "  Edit shell/so101_env.sh or export ROBOT_PORT=/dev/tty.usbmodemXXXX" >&2
    exit 1
  fi
}

require_teleop_port() {
  if [[ -z "${TELEOP_PORT}" ]]; then
    echo "[error] TELEOP_PORT is not set (required for teleop/record)." >&2
    echo "  Edit shell/so101_env.sh or export TELEOP_PORT=/dev/tty.usbmodemYYYY" >&2
    exit 1
  fi
}

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "$@"
  else
    # Do not wrap with `env VAR=...` — that drops ROBOT_PORT and other exports.
    conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
  fi
}

# LeRobot upstream script module (registers opencv_crop via local_lerobot_entry.py first).
run_lerobot_script() {
  local module="$1"
  shift
  run_python "${ROOT}/local_lerobot_entry.py" "${module}" "$@"
}
