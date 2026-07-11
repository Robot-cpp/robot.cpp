#!/usr/bin/env bash
# Shared SO101 defaults for calibrate / teleoperate / record / client scripts.
# Edit the values below for your machine (serial ports, camera, server, task, etc.).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_CPP_ROOT="$(cd "${ROOT}/../.." && pwd)"

export PYTHONPATH="${ROBOT_CPP_ROOT}:${ROOT}:${ROBOT_CPP_ROOT}/robot_client/python:${ROBOT_CPP_ROOT}/robot_client${PYTHONPATH:+:${PYTHONPATH}}"

# --- Robot serial ports ---
export ROBOT_PORT="${ROBOT_PORT:-?ROBOT_PORT must be set}"
export TELEOP_PORT="${TELEOP_PORT:-?ROBOT_PORT must be set}"

export ROBOT_TYPE="so101_follower"
export TELEOP_TYPE="so101_leader"
export ROBOT_USE_DEGREES="true"

# --- Camera ---
export CAMERA_TYPE="realsense"  # iphone / realsense
export CAMERA_RESIZE_WIDTH=224
export CAMERA_RESIZE_HEIGHT=224
export CAMERA_KEY="camera1"
export MODEL_IMAGE_NAME="observation.images.camera1"

case "${CAMERA_TYPE}" in
  iphone)
    export CAMERA_DRIVER="opencv_crop"
    export CAMERA_INDEX=0
    export CAMERA_WIDTH=1280
    export CAMERA_HEIGHT=720
    export CAMERA_BACKEND="AVFOUNDATION"
    export CAMERA_FPS=30
    export CAMERA_WARMUP_S=5
    ;;
  realsense)
    export CAMERA_DRIVER="realsense"
    export REALSENSE_SERIAL="${REALSENSE_SERIAL:-?REALSENSE_SERIAL must be set}"  # 141722072266
    export CAMERA_WIDTH=640
    export CAMERA_HEIGHT=480
    export CAMERA_FPS=30
    if [[ "$(uname -s)" == "Darwin" ]]; then
      export CAMERA_WARMUP_S=15
      export REALSENSE_AUTO_PROFILE=1
    else
      export CAMERA_BACKEND="DSHOW"
      export CAMERA_WARMUP_S=5
    fi
    ;;
  *)
    echo "unsupported CAMERA_TYPE=${CAMERA_TYPE} (use iphone or realsense)" >&2
    exit 1
    ;;
esac

# --- Inference client (python -m eval.lerobot_so101.run_sync) ---
export ROBOT_PLATFORM="lerobot_so101"
export SERVER="${SERVER:-127.0.0.1:5555}"
export TASK="${TASK:-grab the block.}"
export FPS=25


require_teleop_port() {
  if [[ -z "${TELEOP_PORT}" ]]; then
    echo "[error] TELEOP_PORT is not set (required for teleop/record/leader calibrate)." >&2
    echo "  Edit script/shell/so101_env.sh or export TELEOP_PORT=/dev/tty.usbmodemYYYY" >&2
    exit 1
  fi
}

_realsense_python() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "${CONDA_PREFIX}/bin/python"
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    echo python
  fi
}

_build_robot_cameras_json() {
  local py_cmd=(python)
  if [[ "$(id -u)" -eq 0 ]]; then
    py_cmd=("$(_realsense_python)")
  fi
  "${py_cmd[@]}" "${ROOT}/script/shell/build_robot_cameras.py"
}

_verify_python_imports() {
  local py="$(_realsense_python)"
  if ! "${py}" -c "import eval.lerobot_so101.utils.robot" 2>/dev/null; then
    echo "[error] Cannot import eval.* (No module named 'eval')." >&2
    echo "  Repo root: ${ROBOT_CPP_ROOT}" >&2
    echo "  PYTHONPATH=${PYTHONPATH}" >&2
    echo "  Run from ${ROOT}: ./test/run_camera_test.sh" >&2
    exit 1
  fi
  if [[ "${CAMERA_DRIVER:-}" == "realsense" && "$(uname -s)" == "Darwin" ]]; then
    if ! env -u DYLD_LIBRARY_PATH "${py}" -c "import pyrealsense2 as rs" 2>/dev/null; then
      echo "[error] pyrealsense2 import failed." >&2
      echo "  Fix: bash ${ROOT}/script/shell/fix_macos_pyrealsense_lib.sh" >&2
      echo "  Setup: bash ${ROOT}/script/shell/setup_macos_realsense.sh" >&2
      exit 1
    fi
  fi
}

if [[ -n "${ROBOT_CAMERAS:-}" ]]; then
  if ! python -c "import json, os; json.loads(os.environ['ROBOT_CAMERAS'])" 2>/dev/null; then
    echo "[warn] Invalid ROBOT_CAMERAS JSON; rebuilding from CAMERA_TYPE settings." >&2
    unset ROBOT_CAMERAS
  fi
fi

if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  export ROBOT_CAMERAS="$(_build_robot_cameras_json)"
fi
export ROBOT_CAMERAS="${ROBOT_CAMERAS//$'\n'/}"

run_python() {
  _verify_python_imports

  if [[ "$(uname -s)" == "Darwin" && "${CAMERA_DRIVER:-}" == "realsense" && "$(id -u)" -ne 0 && "${REALSENSE_SUDO:-1}" != "0" ]]; then
    echo "[so101] macOS RealSense needs a root shell (sudo python can segfault with conda)." >&2
    echo "[so101] Run:" >&2
    echo "  sudo -s" >&2
    echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate lerobot-demo" >&2
    echo "  cd \"${ROOT}\"" >&2
    echo "  export REALSENSE_SUDO=0 CAMERA_TYPE=realsense REALSENSE_SERIAL=${REALSENSE_SERIAL}" >&2
    echo "  FRAMES=5 ./test/run_camera_test.sh --preview" >&2
    exit 1
  fi

  if [[ "$(id -u)" -eq 0 ]]; then
    "$(_realsense_python)" "$@"
    return
  fi

  python "$@"
}

run_lerobot_script() {
  run_python -m "$@"
}
