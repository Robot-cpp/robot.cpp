#!/usr/bin/env bash
set -e

# ====== change these if needed ======
VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_metal}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5569}"
THREADS="${THREADS:-8}"
PROMPT="${PROMPT:-grab the block.}"
IMAGE_WIDTH="${IMAGE_WIDTH:-224}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-224}"
STATE_DIM="${STATE_DIM:-6}"
WARMUP="${WARMUP:-5}"
LOOPS="${LOOPS:-50}"
DTYPE="${DTYPE:-f32}"

PYTHON="${PYTHON:-python3}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${VLA_CPP_ROOT}/debug/artifacts/robot_server_latency_metal}"
# ====================================

LAUNCH_SHELL="${VLA_CPP_ROOT}/robot_server/shell/launch_robot_server_mac_metal.sh"
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


echo "== configure =="
"${CMAKE_BIN}" -S "${VLA_CPP_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Apple \
    -DGGML_OPENMP=OFF \
    -DGGML_METAL=ON \
    -DVLACPP_BUILD_SMOLVLA=ON \
    -DVLACPP_BUILD_TESTS=ON

echo "== build =="
"${CMAKE_BIN}" --build "${BUILD_DIR}" \
    --target model-server \
    -j8

echo "== prepare outputs =="
mkdir -p "${ARTIFACT_DIR}"
rm -f "${RESULT_TSV}"

echo "== launch server =="
VLA_CPP_ROOT="${VLA_CPP_ROOT}" \
BUILD_DIR="${BUILD_DIR}" \
GGUF_DIR="${GGUF_DIR}" \
DTYPE="${DTYPE}" \
LLM_GGUF="${LLM_GGUF:-}" \
VISION_GGUF="${VISION_GGUF:-}" \
STATE_PROJ_GGUF="${STATE_PROJ_GGUF:-}" \
ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF:-}" \
HOST="${HOST}" \
PORT="${PORT}" \
THREADS="${THREADS}" \
TASK="${PROMPT}" \
NOISE_MODE="gaussian" \
NOISE_SEED="-1" \
SKIP_BUILD="1" \
bash "${LAUNCH_SHELL}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "== wait server =="
sleep 3

echo "== run latency benchmark =="
"${PYTHON}" "${BENCHMARK_SCRIPT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --width "${IMAGE_WIDTH}" \
    --height "${IMAGE_HEIGHT}" \
    --state-dim "${STATE_DIM}" \
    --prompt "${PROMPT}" \
    --warmup "${WARMUP}" \
    --loops "${LOOPS}" \
    --result-tsv "${RESULT_TSV}"

echo "== done =="
echo "result tsv: ${RESULT_TSV}"
echo "server log: ${SERVER_LOG}"
