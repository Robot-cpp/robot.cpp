#!/usr/bin/env bash
# Repeated predict loop against smolvla-server (TCP). Start the server first:
#   bash robot_server/shell/launch_robot_server_mac_cpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_CPP_ROOT="$(cd "${ROOT}/../../../.." && pwd)"

if [[ -f "${VLA_CPP_ROOT}/local_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${VLA_CPP_ROOT}/local_env.sh"
fi

CONDA_ENV="${CONDA_ENV:-lerobot-py312}"
export PYTHONPATH="${LEROBOT_SO101_ROOT:-${ROOT}}/src:${ROOT}/..${PYTHONPATH:+:${PYTHONPATH}}"

HOST="${SMOLVLA_HOST:-${HOST:-127.0.0.1}}"
PORT="${SMOLVLA_PORT:-${PORT:-5555}}"
TASK="${TASK:-grab the block.}"
LOOP_TEST_LOOPS="${LOOP_TEST_LOOPS:-50}"  # (0 = run robot client until Ctrl-C).
WARMUP="${WARMUP:-5}"
SLEEP_MS="${SLEEP_MS:-0}"
QUIET="${QUIET:-0}"
VERBOSE_TIMING="${VERBOSE_TIMING:-1}"
RANDOM_IMAGE="${RANDOM_IMAGE:-0}"

# Same default state as bitvla loop_test.sh (6-D SO101).
STATE_CSV="${STATE_CSV:-0.5479121208190918,-0.12224312126636505,0.7171958684921265,0.39473605155944824,-0.8116453289985657,0.9512447118759155}"
IMAGE_PATH="${IMAGE_PATH:-}"

ARGS=(
  "${ROOT}/test/test_loop_predict.py"
  --host "${HOST}"
  --port "${PORT}"
  --task "${TASK}"
  --state "${STATE_CSV}"
  --warmup "${WARMUP}"
  --loops "${LOOP_TEST_LOOPS}"
  --sleep-ms "${SLEEP_MS}"
)

if [[ "${RANDOM_IMAGE}" == "1" || -z "${IMAGE_PATH}" ]]; then
  echo "[loop-test] image=random host=${HOST} port=${PORT} loops=${LOOP_TEST_LOOPS} warmup=${WARMUP}"
  ARGS+=(--random-image)
else
  if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "[error] image not found: ${IMAGE_PATH}" >&2
    exit 1
  fi
  echo "[loop-test] image=${IMAGE_PATH} host=${HOST} port=${PORT} loops=${LOOP_TEST_LOOPS} warmup=${WARMUP}"
  ARGS+=(--image "${IMAGE_PATH}")
fi

if [[ "${QUIET}" == "1" ]]; then
  ARGS+=(--quiet)
fi
if [[ "${VERBOSE_TIMING}" == "1" ]]; then
  ARGS+=(--verbose-timing)
fi

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    python "${ARGS[@]}" "$@"
  else
    conda run -n "${CONDA_ENV}" env PYTHONPATH="${PYTHONPATH}" python "${ARGS[@]}" "$@"
  fi
}

run_python "$@"
