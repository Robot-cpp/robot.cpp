#!/usr/bin/env bash
set -e

# ====== change these if needed ======
VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must be set}"
BACKEND="${BACKEND:-metal}"
if [ "${1:-}" = "cpu" ] || [ "${1:-}" = "metal" ]; then
    BACKEND="$1"
    shift
fi
MODEL_TYPE="${1:-${MODEL_TYPE:-smolvla}}"

case "${BACKEND}" in
    cpu)
        BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_cpu}"
        ARTIFACT_DIR="${ARTIFACT_DIR:-${VLA_CPP_ROOT}/debug/artifacts/robot_server_latency}"
        LAUNCH_SHELL="${VLA_CPP_ROOT}/robot_server/shell/launch_robot_server_mac_cpu.sh"
        ;;
    metal)
        BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_metal}"
        ARTIFACT_DIR="${ARTIFACT_DIR:-${VLA_CPP_ROOT}/debug/artifacts/robot_server_latency_metal}"
        LAUNCH_SHELL="${VLA_CPP_ROOT}/robot_server/shell/launch_robot_server_mac_metal.sh"
        ;;
    *)
        echo "unsupported BACKEND=${BACKEND}" >&2
        exit 1
        ;;
esac

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5569}"
THREADS="${THREADS:-8}"
PROMPT="${PROMPT:-grab the block.}"
IMAGE_NAME="${IMAGE_NAME:-observation.images.front}"
IMAGE_WIDTH="${IMAGE_WIDTH:-224}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-224}"
STATE_DIM="${STATE_DIM:-6}"
WARMUP="${WARMUP:-5}"
LOOPS="${LOOPS:-50}"
DTYPE="${DTYPE:-f32}"

PYTHON="${PYTHON:-python3}"
# ====================================

BENCHMARK_SCRIPT="${VLA_CPP_ROOT}/robot_server/test/benchmark_latency.py"

SERVER_LOG="${ARTIFACT_DIR}/server.log"
RESULT_TSV="${ARTIFACT_DIR}/benchmark_server.tsv"

# once exit the shell script, make sure to kill the server process if it's still running
cleanup() {
    if [ -n "${SERVER_PID}" ]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
SERVER_PID=""
trap cleanup EXIT

echo "== prepare outputs =="
mkdir -p "${ARTIFACT_DIR}"
rm -f "${RESULT_TSV}"

echo "== launch server =="
VLA_CPP_ROOT="${VLA_CPP_ROOT}" \
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
THREADS="${THREADS}" \
TASK="${PROMPT}" \
NOISE_MODE="gaussian" \
NOISE_SEED="-1" \
bash "${LAUNCH_SHELL}" "${MODEL_TYPE}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "== wait server =="
sleep 5

echo "== run latency benchmark =="
"${PYTHON}" "${BENCHMARK_SCRIPT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --image-name "${IMAGE_NAME}" \
    --width "${IMAGE_WIDTH}" \
    --height "${IMAGE_HEIGHT}" \
    --state-dim "${STATE_DIM}" \
    --prompt "${PROMPT}" \
    --warmup "${WARMUP}" \
    --loops "${LOOPS}" \
    --result-tsv "${RESULT_TSV}"

echo "== done =="
echo "backend: ${BACKEND}"
echo "result tsv: ${RESULT_TSV}"
echo "server log: ${SERVER_LOG}"
