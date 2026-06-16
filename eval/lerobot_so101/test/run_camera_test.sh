#!/usr/bin/env bash
# Camera-only smoke test; same defaults as shell/run_robot_client.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/shell/so101_env.sh"

FRAMES="${FRAMES:-0}"
PREVIEW="${PREVIEW:-1}"

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

if [[ -n "${ROBOT_CAMERAS:-}" ]]; then
  echo "[camera-test] camera_key=${CAMERA_KEY} index=${CAMERA_INDEX} preview=${PREVIEW} frames=${FRAMES}"
  ARGS+=(--robot-cameras "${ROBOT_CAMERAS}")
else
  case "${1:-}" in
    --list-cameras|--probe)
      echo "[camera-test] mode=${1} camera_key=${CAMERA_KEY}"
      ;;
    *)
      echo "[error] ROBOT_CAMERAS is not set" >&2
      exit 1
      ;;
  esac
fi

run_python "${ARGS[@]}" "$@"
