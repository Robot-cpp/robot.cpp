#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_CPP_ROOT="${VLA_CPP_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

MODEL_TYPE="${MODEL_TYPE:-smolvla}"
DTYPE="${DTYPE:-f16}"
BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build-linux-cpu}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
THREADS="${THREADS:-8}"
TASK="${TASK:-grab the block.}"
N_BATCH="${N_BATCH:-512}"
N_CTX="${N_CTX:-2048}"
NOISE_MODE="${NOISE_MODE:-gaussian}"
NOISE_SEED="${NOISE_SEED:--1}"
VERBOSITY="${VERBOSITY:-0}"

SKIP_BUILD="${SKIP_BUILD:-0}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
GGML_NATIVE="${GGML_NATIVE:-OFF}"
GGML_BLAS="${GGML_BLAS:-OFF}"
GGML_OPENMP="${GGML_OPENMP:-OFF}"

if [ -z "${GGUF_DIR:-}" ]; then
    case "${MODEL_TYPE}" in
        smolvla)
            GGUF_DIR="${VLA_CPP_ROOT}/ckpts/smolvla/gguf-${DTYPE}"
            ;;
        pi0)
            GGUF_DIR="${VLA_CPP_ROOT}/ckpts/pi0-libero-finetuned-v044/vlacpp-split"
            ;;
        *)
            echo "unsupported MODEL_TYPE=${MODEL_TYPE}" >&2
            exit 1
            ;;
    esac
fi

SERVER_BIN="${BUILD_DIR}/bin/model-server"

if [ "${SKIP_BUILD}" != "1" ]; then
    CMAKE_FLAGS=(
        -S "${VLA_CPP_ROOT}"
        -B "${BUILD_DIR}"
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_NATIVE="${GGML_NATIVE}"
        -DGGML_BLAS="${GGML_BLAS}"
        -DGGML_OPENMP="${GGML_OPENMP}"
        -DGGML_CUDA=OFF
        -DGGML_METAL=OFF
        -DBUILD_ROBOT_SERVER=ON
    )
    if [ -n "${GGML_BLAS_VENDOR:-}" ]; then
        CMAKE_FLAGS+=(-DGGML_BLAS_VENDOR="${GGML_BLAS_VENDOR}")
    fi

    echo "== configure =="
    "${CMAKE_BIN}" "${CMAKE_FLAGS[@]}"

    echo "== build =="
    "${CMAKE_BIN}" --build "${BUILD_DIR}" --target model-server -j8
fi

case "${MODEL_TYPE}" in
    smolvla)
        LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/smolvla-llm-${DTYPE}.gguf}"
        MMPROJ_GGUF="${MMPROJ_GGUF:-${VISION_GGUF:-${GGUF_DIR}/mmproj-smolvla-${DTYPE}.gguf}}"
        STATE_PROJ_GGUF="${STATE_PROJ_GGUF:-${GGUF_DIR}/state-proj-smolvla-${DTYPE}.gguf}"
        ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF:-${GGUF_DIR}/action-expert-smolvla-${DTYPE}.gguf}"
        MODEL_ARGS=(
            --model-type smolvla
            --llm "${LLM_GGUF}"
            --mmproj "${MMPROJ_GGUF}"
            --state-proj "${STATE_PROJ_GGUF}"
            --action-expert "${ACTION_EXPERT_GGUF}"
        )
        ;;
    pi0)
        MODEL_BASENAME="${MODEL_BASENAME:-vlacpp-pi0-libero-finetuned-v044}"
        VIT_GGUF="${VIT_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.vit.gguf}"
        MMPROJ_GGUF="${MMPROJ_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.mmproj.gguf}"
        LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.llm.gguf}"
        TOKENIZER_GGUF="${TOKENIZER_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.tokenizer.gguf}"
        STATE_GGUF="${STATE_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.state.gguf}"
        ACTION_DECODER_GGUF="${ACTION_DECODER_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.action_decoder.gguf}"
        MODEL_ARGS=(
            --model-type pi0
            --vit "${VIT_GGUF}"
            --mmproj "${MMPROJ_GGUF}"
            --llm "${LLM_GGUF}"
            --tokenizer "${TOKENIZER_GGUF}"
            --state-gguf "${STATE_GGUF}"
            --action-decoder "${ACTION_DECODER_GGUF}"
        )
        ;;
    *)
        echo "unsupported MODEL_TYPE=${MODEL_TYPE}" >&2
        exit 1
        ;;
esac

echo "== launch server =="
echo "model_type: ${MODEL_TYPE}"
echo "host: ${HOST}"
echo "port: ${PORT}"
echo "gguf_dir: ${GGUF_DIR}"

exec "${SERVER_BIN}" \
    "${MODEL_ARGS[@]}" \
    --task "${TASK}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --threads "${THREADS}" \
    --n-batch "${N_BATCH}" \
    --n-ctx "${N_CTX}" \
    --noise-mode "${NOISE_MODE}" \
    --noise-seed "${NOISE_SEED}" \
    --verbosity "${VERBOSITY}"
