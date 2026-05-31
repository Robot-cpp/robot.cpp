#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-vlacpp-lerobot}"

ROBOT_PORT="${ROBOT_PORT:-/dev/tty.usbmodem5B3E1195731}"
ROBOT_CAMERAS="${ROBOT_CAMERAS:-{\"camera1\":{\"type\":\"opencv_crop\",\"index_or_path\":0,\"width\":1280,\"height\":720,\"fps\":30,\"backend\":\"AVFOUNDATION\",\"resize_width\":224,\"resize_height\":224,\"center_crop_square_before_resize\":true,\"warmup_s\":5}}}"
TASK="${TASK:-grab the block}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
PRETRAINED="${PRETRAINED_NAME_OR_PATH:-}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
POLICY_VLM="${POLICY_VLM_MODEL_NAME:-}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
FPS="${FPS:-10}"
CHUNK_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.2}"
AGGREGATE_FN="${AGGREGATE_FN_NAME:-weighted_average}"
POLICY_FORCE_FP32="${POLICY_FORCE_FP32:-true}"

if [[ -z "${PRETRAINED}" ]]; then
  echo "[error] Set PRETRAINED_NAME_OR_PATH" >&2
  exit 1
fi

EXTRA_VLM=()
if [[ -n "${POLICY_VLM}" ]]; then
  EXTRA_VLM=(--policy_vlm_model_name="${POLICY_VLM}")
fi

echo "[async-client] server=${SERVER_ADDRESS} policy=${POLICY_TYPE}"

conda run -n "${CONDA_ENV}" vlacpp-async-robot-client \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --task="${TASK}" \
  --server_address="${SERVER_ADDRESS}" \
  --policy_type="${POLICY_TYPE}" \
  --pretrained_name_or_path="${PRETRAINED}" \
  --policy_device="${POLICY_DEVICE}" \
  --policy_force_fp32="${POLICY_FORCE_FP32}" \
  "${EXTRA_VLM[@]}" \
  --client_device="${CLIENT_DEVICE}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --fps="${FPS}" \
  --chunk_size_threshold="${CHUNK_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN}"
