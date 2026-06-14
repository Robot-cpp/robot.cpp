#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_CPP_ROOT="${VLA_CPP_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ====== change these if needed ======
GGUF_DIR="${GGUF_DIR:-${VLA_CPP_ROOT}/ckpts/pi0-libero-finetuned-v044/vlacpp-split}"
BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_pi0_server_cpu}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
THREADS="${THREADS:-8}"
TASK="${TASK:-pick up the fork}"
NOISE_SEED="${NOISE_SEED:--1}"

SKIP_BUILD="${SKIP_BUILD:-0}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
MODEL_BASENAME="${MODEL_BASENAME:-vlacpp-pi0-libero-finetuned-v044}"
# ====================================

VIT_GGUF="${VIT_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.vit.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.mmproj.gguf}"
LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.llm.gguf}"
TOKENIZER_GGUF="${TOKENIZER_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.tokenizer.gguf}"
STATE_GGUF="${STATE_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.state.gguf}"
ACTION_DECODER_GGUF="${ACTION_DECODER_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.action_decoder.gguf}"
SERVER_BIN="${BUILD_DIR}/bin/model-server"

if [ "${SKIP_BUILD}" != "1" ]; then
    echo "== configure =="
    "${CMAKE_BIN}" -S "${VLA_CPP_ROOT}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_METAL=OFF \
        -DVLACPP_BUILD_ROBOT_SERVER=ON

    echo "== build =="
    "${CMAKE_BIN}" --build "${BUILD_DIR}" \
        --target model-server \
        -j8
fi

echo "== launch pi0 server =="
echo "host: ${HOST}"
echo "port: ${PORT}"
echo "task: ${TASK}"
echo "gguf_dir: ${GGUF_DIR}"

exec "${SERVER_BIN}" \
    --model-type pi0 \
    --vit "${VIT_GGUF}" \
    --mmproj "${MMPROJ_GGUF}" \
    --llm "${LLM_GGUF}" \
    --tokenizer "${TOKENIZER_GGUF}" \
    --state-gguf "${STATE_GGUF}" \
    --action-decoder "${ACTION_DECODER_GGUF}" \
    --task "${TASK}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --threads "${THREADS}" \
    --noise-seed "${NOISE_SEED}" \
    --verbosity 0
