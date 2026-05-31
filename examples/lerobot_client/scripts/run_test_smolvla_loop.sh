#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV_NAME:-vlacpp-lerobot}"

CKPT_PATH="${CKPT_PATH:-${ROOT}/checkpoint/smolvla_450M_50k_grab_block_70/pretrained_model}"
TEST_IMAGE_PATH="${TEST_IMAGE_PATH:-${ROOT}/tools/fixtures/test_image.jpg}"
TEST_PY="${ROOT}/tools/test_smolvla_loop.py"

DEVICE="${DEVICE:-cpu}"
LOOPS="${LOOPS:-50}"
WARMUP="${WARMUP:-1}"
SLEEP_MS="${SLEEP_MS:-0}"
TASK="${TASK:-grab the block}"
FORCE_FP32="${FORCE_FP32:-1}"
IMG_H="${IMG_H:-224}"
IMG_W="${IMG_W:-224}"

echo "[smolvla-loop] ckpt=${CKPT_PATH} device=${DEVICE} loops=${LOOPS}"

FP32_FLAG="--force-fp32"
if [[ "${FORCE_FP32}" == "0" ]]; then
  FP32_FLAG="--no-force-fp32"
fi

conda run -n "${CONDA_ENV}" python "${TEST_PY}" \
  --pretrained-path "${CKPT_PATH}" \
  --device "${DEVICE}" \
  --image-path "${TEST_IMAGE_PATH}" \
  --task "${TASK}" \
  --loops "${LOOPS}" \
  --warmup "${WARMUP}" \
  --sleep-ms "${SLEEP_MS}" \
  --image-h "${IMG_H}" \
  --image-w "${IMG_W}" \
  "${FP32_FLAG}"
