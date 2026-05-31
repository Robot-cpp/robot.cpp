#!/usr/bin/env bash
# Shared SO101 defaults for calibrate / teleoperate / record scripts.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLA_CPP_ROOT="$(cd "${ROOT}/../../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-vlacpp-lerobot}"

ROBOT_PORT="${ROBOT_PORT:-/dev/tty.usbmodem5B3E1195731}"
TELEOP_PORT="${TELEOP_PORT:-/dev/tty.usbmodem5B3E1198201}"

CAMERAS_JSON="${CAMERAS_JSON:-${ROOT}/configs/cameras/front.json}"

if [[ ! -f "${CAMERAS_JSON}" ]]; then
  echo "[error] camera config not found: ${CAMERAS_JSON}" >&2
  exit 1
fi

ROBOT_CAMERAS="$(tr -d '\n' < "${CAMERAS_JSON}")"

run_lerobot() {
  conda run -n "${CONDA_ENV}" "$@"
}
