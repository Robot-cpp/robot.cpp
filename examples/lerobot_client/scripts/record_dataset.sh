#!/usr/bin/env bash
set -euo pipefail

# Record SO101 dataset with leader teleop and front camera.
# Usage: ./scripts/record_dataset.sh

source "$(dirname "$0")/lib/so101_env.sh"

DATASET_REPO_ID="YOUR_HF_USER/my_so101_dataset"
DATASET_ROOT="${ROOT}/data"
DATASET_PUSH_TO_HUB=false
NUM_EPISODES=50
SINGLE_TASK="grab the block"
DATASET_FPS=30
EPISODE_TIME_S=20
RESET_TIME_S=10
STREAMING_ENCODING=true
ENCODER_THREADS=2
DISPLAY_DATA=true

mkdir -p "${DATASET_ROOT}"

echo "[record] repo_id=${DATASET_REPO_ID} root=${DATASET_ROOT} episodes=${NUM_EPISODES}"

run_lerobot lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=so101_leader \
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
