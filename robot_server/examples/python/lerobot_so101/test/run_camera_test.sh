#!/usr/bin/env bash
# Camera-only smoke test; same defaults as scripts/run_robot_client.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_CPP_ROOT="$(cd "${ROOT}/../../../.." && pwd)"

if [[ -f "${VLA_CPP_ROOT}/local_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${VLA_CPP_ROOT}/local_env.sh"
fi

CONDA_ENV="${CONDA_ENV:-lerobot-py312}"
export PYTHONPATH="${LEROBOT_SO101_ROOT:-${ROOT}}/src:${LEROBOT_SO101_ROOT:-${ROOT}}/src/lerobot_camera_crop${PYTHONPATH:+:${PYTHONPATH}}"

CAMERA_KEY="${CAMERA_KEY:-camera1}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
FRAMES="${FRAMES:-0}"
PREVIEW="${PREVIEW:-1}"
CAMERAS_JSON="${CAMERAS_JSON:-${ROOT}/configs/cameras/front.json}"

ARGS=(
  "${ROOT}/test/test_camera.py"
  --camera-key "${CAMERA_KEY}"
  --frames "${FRAMES}"
)

if [[ -n "${CAMERA_INDEX}" ]]; then
  ARGS+=(--camera-index "${CAMERA_INDEX}")
fi

case "${1:-}" in
  --list-cameras|--probe)
    PREVIEW=0
    ;;
esac

if [[ "${PREVIEW}" != "0" ]]; then
  ARGS+=(--preview)
else
  ARGS+=(--no-preview)
fi

if [[ -n "${CAMERAS_JSON}" ]]; then
  if [[ ! -f "${CAMERAS_JSON}" ]]; then
    echo "[error] camera config not found: ${CAMERAS_JSON}" >&2
    exit 1
  fi
  echo "[camera-test] cameras_json=${CAMERAS_JSON} camera_key=${CAMERA_KEY} index=${CAMERA_INDEX} preview=${PREVIEW} frames=${FRAMES}"
  ARGS+=(--cameras-json "${CAMERAS_JSON}")
else
  case "${1:-}" in
    --list-cameras|--probe)
      echo "[camera-test] mode=${1} camera_key=${CAMERA_KEY}"
      ;;
    *)
      echo "[error] CAMERAS_JSON is not set" >&2
      exit 1
      ;;
  esac
fi

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "${ARGS[@]}" "$@"
  else
    conda run -n "${CONDA_ENV}" env PYTHONPATH="${PYTHONPATH}" python "${ARGS[@]}" "$@"
  fi
}

run_python "$@"
