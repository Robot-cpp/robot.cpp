#!/usr/bin/env bash
# Shared SO101 defaults for calibrate / teleoperate / record / client scripts.
# Serial ports default from configs/robot/*.yaml (override with ROBOT_PORT / TELEOP_PORT).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLA_CPP_ROOT="$(cd "${ROOT}/../../../.." && pwd)"

if [[ -f "${VLA_CPP_ROOT}/local_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${VLA_CPP_ROOT}/local_env.sh"
fi

CONDA_ENV="${CONDA_ENV:-lerobot-py312}"

FOLLOWER_ROBOT_CONFIG="${FOLLOWER_ROBOT_CONFIG:-${ROOT}/configs/robot/so101_follower.yaml}"
LEADER_ROBOT_CONFIG="${LEADER_ROBOT_CONFIG:-${ROOT}/configs/robot/so101_leader.yaml}"

read_robot_config_port() {
  local config_file="$1"
  if [[ ! -f "${config_file}" ]]; then
    echo "[error] robot config not found: ${config_file}" >&2
    return 1
  fi
  local port
  port="$(grep -E '^port:[[:space:]]*' "${config_file}" | sed -E 's/^port:[[:space:]]*//' | head -1)"
  if [[ -z "${port}" ]]; then
    echo "[error] missing port: in ${config_file}" >&2
    return 1
  fi
  printf '%s' "${port}"
}

if [[ -z "${ROBOT_PORT:-}" ]]; then
  ROBOT_PORT="$(read_robot_config_port "${FOLLOWER_ROBOT_CONFIG}")"
fi
if [[ -z "${TELEOP_PORT:-}" ]]; then
  TELEOP_PORT="$(read_robot_config_port "${LEADER_ROBOT_CONFIG}")"
fi

CAMERAS_JSON="${CAMERAS_JSON:-${ROOT}/configs/cameras/front.json}"
CAMERA_KEY="${CAMERA_KEY:-camera1}"

if [[ ! -f "${CAMERAS_JSON}" ]]; then
  echo "[error] camera config not found: ${CAMERAS_JSON}" >&2
  exit 1
fi

ROBOT_CAMERAS="$(tr -d '\n' < "${CAMERAS_JSON}")"

require_robot_port() {
  if [[ -z "${ROBOT_PORT}" ]]; then
    echo "[error] ROBOT_PORT is not set." >&2
    echo "  Set port in ${FOLLOWER_ROBOT_CONFIG} or export ROBOT_PORT=/dev/tty.usbmodemXXXX" >&2
    exit 1
  fi
}

require_teleop_port() {
  if [[ -z "${TELEOP_PORT}" ]]; then
    echo "[error] TELEOP_PORT is not set (required for teleop/record)." >&2
    echo "  Set port in ${LEADER_ROBOT_CONFIG} or export TELEOP_PORT=/dev/tty.usbmodemYYYY" >&2
    exit 1
  fi
}

run_lerobot() {
  local runner="${ROOT}/scripts/lib/run_lerobot_cli.py"
  if [[ ! -f "${runner}" ]]; then
    echo "[error] missing lerobot runner: ${runner}" >&2
    exit 1
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "${runner}" "$@"
  else
    conda run -n "${CONDA_ENV}" python "${runner}" "$@"
  fi
}
