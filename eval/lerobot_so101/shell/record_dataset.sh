#!/usr/bin/env bash
set -euo pipefail

# Record SO101 dataset with leader teleop and front camera.
# Usage: ./shell/local_record_dataset.sh

source "$(dirname "$0")/so101_env.sh"

require_robot_port
require_teleop_port

mkdir -p "${DATASET_ROOT}"

echo "[record] repo_id=${DATASET_REPO_ID} root=${DATASET_ROOT} episodes=${NUM_EPISODES}"

run_lerobot_script lerobot.scripts.lerobot_record \
  --robot.type="${ROBOT_TYPE}" \
  --robot.port="${ROBOT_PORT}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type="${TELEOP_TYPE}" \
  --teleop.port="${TELEOP_PORT}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.push_to_hub="${DATASET_PUSH_TO_HUB}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.streaming_encoding="${STREAMING_ENCODING}" \
  --dataset.encoder_threads="${ENCODER_THREADS}" \
  --display_data="${DISPLAY_DATA}"
