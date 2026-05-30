#!/usr/bin/env bash
set -e

# ====== change these if needed ======
VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_cpu}"


HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
THREADS="${THREADS:-8}"
TASK="${TASK:-grab the block.}"
NOISE_MODE="${NOISE_MODE:-gaussian}"
NOISE_SEED="${NOISE_SEED:--1}"

SKIP_BUILD="${SKIP_BUILD:-0}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
# ====================================

VLM_GGUF="${GGUF_DIR}/smolvla-vlm-f32.gguf"
VISION_GGUF="${GGUF_DIR}/mmproj-smolvla-f32.gguf"
STATE_PROJ_GGUF="${GGUF_DIR}/state-proj-smolvla-f32.gguf"
ACTION_EXPERT_GGUF="${GGUF_DIR}/action-expert-smolvla-f32.gguf"
SERVER_BIN="${BUILD_DIR}/bin/smolvla-server"

if [ "${SKIP_BUILD}" != "1" ]; then
    echo "== configure =="
    "${CMAKE_BIN}" -S "${VLA_CPP_ROOT}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=Apple \
        -DGGML_OPENMP=OFF \
        -DGGML_METAL=OFF \
        -DVLACPP_BUILD_SMOLVLA=ON \
        -DVLACPP_BUILD_TESTS=ON

    echo "== build =="
    "${CMAKE_BIN}" --build "${BUILD_DIR}" \
        --target smolvla-server \
        -j8
fi

echo "== launch server =="
echo "host: ${HOST}"
echo "port: ${PORT}"
echo "task: ${TASK}"

"${SERVER_BIN}" \
    --vlm "${VLM_GGUF}" \
    --mmproj "${VISION_GGUF}" \
    --state-proj "${STATE_PROJ_GGUF}" \
    --action-expert "${ACTION_EXPERT_GGUF}" \
    --task "${TASK}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --threads "${THREADS}" \
    --n-batch 512 \
    --n-ctx 2048 \
    --action-dim 6 \
    --chunk-size 50 \
    --num-steps 10 \
    --noise-mode "${NOISE_MODE}" \
    --noise-seed "${NOISE_SEED}" \
    --verbosity 0
