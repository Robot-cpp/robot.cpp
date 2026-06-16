#!/usr/bin/env bash
set -e

# ====== change these if needed ======
VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must be set}"
MODEL_TYPE="${1:-${MODEL_TYPE:-smolvla}}"
BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_cpu}"


HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
THREADS="${THREADS:-8}"
TASK="${TASK:-grab the block.}"
NOISE_MODE="${NOISE_MODE:-gaussian}"
NOISE_SEED="${NOISE_SEED:--1}"

SKIP_BUILD="${SKIP_BUILD:-0}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
DTYPE="${DTYPE:-f32}"
# ====================================

SERVER_BIN="${BUILD_DIR}/bin/model-server"

if [ "${SKIP_BUILD}" != "1" ]; then
    echo "== configure =="
    "${CMAKE_BIN}" -S "${VLA_CPP_ROOT}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=Apple \
        -DGGML_OPENMP=OFF \
        -DGGML_METAL=OFF \
        -DVLACPP_BUILD_ROBOT_SERVER=ON

    echo "== build =="
    "${CMAKE_BIN}" --build "${BUILD_DIR}" \
        --target model-server \
        -j8
fi

case "${MODEL_TYPE}" in
    smolvla)
        LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/smolvla-llm-${DTYPE}.gguf}"
        VISION_GGUF="${VISION_GGUF:-${GGUF_DIR}/mmproj-smolvla-${DTYPE}.gguf}"
        STATE_PROJ_GGUF="${STATE_PROJ_GGUF:-${GGUF_DIR}/state-proj-smolvla-${DTYPE}.gguf}"
        ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF:-${GGUF_DIR}/action-expert-smolvla-${DTYPE}.gguf}"
        MODEL_ARGS=(
            --model-type smolvla
            --llm "${LLM_GGUF}"
            --mmproj "${VISION_GGUF}"
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
echo "task: ${TASK}"

exec "${SERVER_BIN}" \
    "${MODEL_ARGS[@]}" \
    --task "${TASK}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --threads "${THREADS}" \
    --n-batch 512 \
    --n-ctx 2048 \
    --noise-mode "${NOISE_MODE}" \
    --noise-seed "${NOISE_SEED}" \
    --verbosity 0
