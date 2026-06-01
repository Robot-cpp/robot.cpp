#!/usr/bin/env bash
# Local paths for vla.cpp SmolVLA server + SO101 client on this machine.
# Usage: source /Volumes/T7/vla.cpp/local_env.sh

export VLA_CPP_ROOT="${VLA_CPP_ROOT:-/Volumes/T7/vla.cpp}"
export GGUF_DIR="${GGUF_DIR:-${VLA_CPP_ROOT}/ckpts/smolvla}"
export BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_cpu}"

export CONDA_ENV="${CONDA_ENV:-lerobot-py312}"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-5555}"
export SERVER="${SERVER:-127.0.0.1:5555}"
export TASK="${TASK:-grab the block.}"

export LEROBOT_SO101_ROOT="${VLA_CPP_ROOT}/robot_server/examples/python/lerobot_so101"
export CAMERAS_JSON="${CAMERAS_JSON:-${LEROBOT_SO101_ROOT}/configs/cameras/front.json}"
export CAMERA_KEY="${CAMERA_KEY:-camera1}"
export CAMERA_INDEX="${CAMERA_INDEX:-0}"
export FPS="${FPS:-25}"
export LOOPS="${LOOPS:-0}"

# Serial ports: default from configs/robot/so101_{follower,leader}.yaml via so101_env.sh
# Optional override: export ROBOT_PORT=... / TELEOP_PORT=...

export PYTHONPATH="${LEROBOT_SO101_ROOT}/src:${LEROBOT_SO101_ROOT}/src/lerobot_camera_crop${PYTHONPATH:+:${PYTHONPATH}}"
export COPYFILE_DISABLE=1
