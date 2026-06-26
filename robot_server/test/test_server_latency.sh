#!/usr/bin/env bash
set -e

# ====== required env / positional args ======
# Usage:
#   ROBOT_CPP_ROOT=/path/to/vla.cpp GGUF_DIR=/path/to/gguf \
#       bash robot_server/test/test_server_latency.sh <model-type> <backend> <test-suite>
#
# Positional args:
#   $1: model-type, e.g. smolvla / pi0
#   $2: backend, e.g. mac-cpu / mac-metal / linux-cpu / linux-cuda
#   $3: test-suite, e.g. smolvla-libero / smolvla-so101 / pi0-libero
ROBOT_CPP_ROOT="${ROBOT_CPP_ROOT:?ROBOT_CPP_ROOT must be set}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must be set}"
MODEL_TYPE="${1:-${MODEL_TYPE:-smolvla}}"
BACKEND="${2:-${BACKEND:-mac-metal}}"
TEST_SUITE="${3:-${TEST_SUITE:-smolvla-so101}}"

# ====== model GGUF overrides ======
case "${MODEL_TYPE}" in
    smolvla)
        # SmolVLA defaults are resolved by robot_server/shell/launch_robot_server_*.sh:
        #   ${GGUF_DIR}/smolvla-llm-f32.gguf
        #   ${GGUF_DIR}/mmproj-smolvla-f32.gguf
        #   ${GGUF_DIR}/state-proj-smolvla-f32.gguf
        #   ${GGUF_DIR}/action-expert-smolvla-f32.gguf
        LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/smolvla-llm-f32.gguf}"
        VISION_GGUF="${VISION_GGUF:-${GGUF_DIR}/mmproj-smolvla-f32.gguf}"
        STATE_PROJ_GGUF="${STATE_PROJ_GGUF:-${GGUF_DIR}/state-proj-smolvla-f32.gguf}"
        ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF:-${GGUF_DIR}/action-expert-smolvla-f32.gguf}"
        ;;
    pi0)
        # pi0 defaults are resolved from MODEL_BASENAME by launch scripts.
        MODEL_BASENAME="${MODEL_BASENAME:-pi0}"
        VIT_GGUF="${VIT_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.vit.gguf}"
        MMPROJ_GGUF="${MMPROJ_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.mmproj.gguf}"
        TOKENIZER_GGUF="${TOKENIZER_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.tokenizer.gguf}"
        STATE_GGUF="${STATE_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.state.gguf}"
        ACTION_DECODER_GGUF="${ACTION_DECODER_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.action_decoder.gguf}"
        LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/${MODEL_BASENAME}.llm.gguf}"
        ;;
    *)
        echo "unsupported MODEL_TYPE=${MODEL_TYPE}" >&2
        exit 1
        ;;
esac

case "${BACKEND}" in
    mac-cpu)
        BUILD_DIR="${BUILD_DIR:-${ROBOT_CPP_ROOT}/build_mac_cpu}"
        ARTIFACT_DIR="${ARTIFACT_DIR:-${ROBOT_CPP_ROOT}/debug/artifacts/robot_server_latency}"
        LAUNCH_SHELL="${ROBOT_CPP_ROOT}/robot_server/shell/launch_robot_server_mac_cpu.sh"
        ;;
    mac-metal)
        BUILD_DIR="${BUILD_DIR:-${ROBOT_CPP_ROOT}/build_mac_metal}"
        ARTIFACT_DIR="${ARTIFACT_DIR:-${ROBOT_CPP_ROOT}/debug/artifacts/robot_server_latency_metal}"
        LAUNCH_SHELL="${ROBOT_CPP_ROOT}/robot_server/shell/launch_robot_server_mac_metal.sh"
        ;;
    linux-cpu)
        BUILD_DIR="${BUILD_DIR:-${ROBOT_CPP_ROOT}/build_linux_cpu}"
        ARTIFACT_DIR="${ARTIFACT_DIR:-${ROBOT_CPP_ROOT}/debug/artifacts/robot_server_latency_linux_cpu}"
        LAUNCH_SHELL="${ROBOT_CPP_ROOT}/robot_server/shell/launch_robot_server_linux_cpu.sh"
        ;;
    linux-cuda)
        BUILD_DIR="${BUILD_DIR:-${ROBOT_CPP_ROOT}/build_linux_cuda}"
        ARTIFACT_DIR="${ARTIFACT_DIR:-${ROBOT_CPP_ROOT}/debug/artifacts/robot_server_latency_linux_cuda}"
        LAUNCH_SHELL="${ROBOT_CPP_ROOT}/robot_server/shell/launch_robot_server_linux_cuda.sh"
        ;;
    *)
        echo "unsupported BACKEND=${BACKEND}" >&2
        exit 1
        ;;
esac

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5569}"
THREADS="${THREADS:-8}"
THREADS_SWEEP="${THREADS_SWEEP:-0}"
THREADS_MIN="${THREADS_MIN:-4}"
THREADS_MAX="${THREADS_MAX:-16}"
THREADS_STEP="${THREADS_STEP:-4}"
PROMPT="${PROMPT:-grab the block.}"
case "${TEST_SUITE}" in
    smolvla-libero)
        DEFAULT_IMAGE_NAMES="observation.images.image,observation.images.image2"
        DEFAULT_IMAGE_WIDTH="256"
        DEFAULT_IMAGE_HEIGHT="256"
        DEFAULT_STATE_DIM="8"
        ;;
    smolvla-so101)
        DEFAULT_IMAGE_NAMES="observation.images.front"
        DEFAULT_IMAGE_WIDTH="224"
        DEFAULT_IMAGE_HEIGHT="224"
        DEFAULT_STATE_DIM="6"
        ;;
    pi0-libero)
        DEFAULT_IMAGE_NAMES="observation.images.image,observation.images.image2"
        DEFAULT_IMAGE_WIDTH="256"
        DEFAULT_IMAGE_HEIGHT="256"
        DEFAULT_STATE_DIM="32"
        ;;
    *)
        echo "unsupported TEST_SUITE=${TEST_SUITE}" >&2
        exit 1
        ;;
esac
IMAGE_NAMES="${IMAGE_NAMES:-${IMAGE_NAME:-${DEFAULT_IMAGE_NAMES}}}"
IMAGE_WIDTH="${IMAGE_WIDTH:-${DEFAULT_IMAGE_WIDTH}}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-${DEFAULT_IMAGE_HEIGHT}}"
STATE_DIM="${STATE_DIM:-${DEFAULT_STATE_DIM}}"
WARMUP="${WARMUP:-5}"
LOOPS="${LOOPS:-100}"
SERVER_WAIT_S="${SERVER_WAIT_S:-120}"
DTYPE="${DTYPE:-f32}"

PYTHON="${PYTHON:-python3}"
# ====================================

BENCHMARK_SCRIPT="${ROBOT_CPP_ROOT}/robot_server/test/benchmark_latency.py"

RESULT_TSV="${ARTIFACT_DIR}/benchmark_server.tsv"
BENCHMARK_IMAGE_ARGS=()
IFS=',' read -r -a IMAGE_NAME_LIST <<< "${IMAGE_NAMES}"
for IMAGE_NAME_ITEM in "${IMAGE_NAME_LIST[@]}"; do
    if [ -n "${IMAGE_NAME_ITEM}" ]; then
        BENCHMARK_IMAGE_ARGS+=(--image-name "${IMAGE_NAME_ITEM}")
    fi
done

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT

run_latency_case() {
    local threads="$1"
    local server_log="${ARTIFACT_DIR}/server_t${threads}.log"

    cleanup

    echo "== launch server (threads=${threads}) =="
    ROBOT_CPP_ROOT="${ROBOT_CPP_ROOT}" \
    BUILD_DIR="${BUILD_DIR}" \
    MODEL_TYPE="${MODEL_TYPE}" \
    GGUF_DIR="${GGUF_DIR}" \
    DTYPE="${DTYPE}" \
    LLM_GGUF="${LLM_GGUF:-}" \
    MMPROJ_GGUF="${MMPROJ_GGUF:-}" \
    VISION_GGUF="${VISION_GGUF:-}" \
    STATE_PROJ_GGUF="${STATE_PROJ_GGUF:-}" \
    ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF:-}" \
    MODEL_BASENAME="${MODEL_BASENAME:-}" \
    VIT_GGUF="${VIT_GGUF:-}" \
    TOKENIZER_GGUF="${TOKENIZER_GGUF:-}" \
    STATE_GGUF="${STATE_GGUF:-}" \
    ACTION_DECODER_GGUF="${ACTION_DECODER_GGUF:-}" \
    HOST="${HOST}" \
    PORT="${PORT}" \
    THREADS="${threads}" \
    TASK="${PROMPT}" \
    NOISE_MODE="gaussian" \
    NOISE_SEED="-1" \
    bash "${LAUNCH_SHELL}" "${MODEL_TYPE}" >"${server_log}" 2>&1 &
    SERVER_PID=$!

    echo "== run latency benchmark (threads=${threads}) =="
    "${PYTHON}" "${BENCHMARK_SCRIPT}" \
        --host "${HOST}" \
        --port "${PORT}" \
        "${BENCHMARK_IMAGE_ARGS[@]}" \
        --width "${IMAGE_WIDTH}" \
        --height "${IMAGE_HEIGHT}" \
        --state-dim "${STATE_DIM}" \
        --prompt "${PROMPT}" \
        --warmup "${WARMUP}" \
        --loops "${LOOPS}" \
        --threads "${threads}" \
        --wait-server-s "${SERVER_WAIT_S}" \
        --result-tsv "${RESULT_TSV}"
}

echo "== prepare outputs =="
mkdir -p "${ARTIFACT_DIR}"
rm -f "${RESULT_TSV}"

if [[ "${THREADS_SWEEP}" == "1" ]]; then
    echo "== thread sweep: ${THREADS_MIN}..${THREADS_MAX} step ${THREADS_STEP} =="
    for ((threads=THREADS_MIN; threads<=THREADS_MAX; threads+=THREADS_STEP)); do
        run_latency_case "${threads}"
    done
else
    run_latency_case "${THREADS}"
fi

echo "== done =="
echo "model_type: ${MODEL_TYPE}"
echo "backend: ${BACKEND}"
echo "test_suite: ${TEST_SUITE}"
echo "image_names: ${IMAGE_NAMES}"
echo "image_size: ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "state_dim: ${STATE_DIM}"
echo "result tsv: ${RESULT_TSV}"
if [[ "${THREADS_SWEEP}" == "1" ]]; then
    echo "server logs: ${ARTIFACT_DIR}/server_t*.log"
else
    echo "server log: ${ARTIFACT_DIR}/server_t${THREADS}.log"
fi
