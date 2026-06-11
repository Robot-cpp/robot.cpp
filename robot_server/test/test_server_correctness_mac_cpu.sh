#!/usr/bin/env bash
set -e

# ====== change these if needed ======
VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
GGUF_DIR="${GGUF_DIR:?GGUF_DIR must be set}"
IMAGE_PATH="${IMAGE_PATH:?IMAGE_PATH must be set}"
BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_cpu}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
THREADS="${THREADS:-8}"
PROMPT="${PROMPT:-grab the block.}"
STATE="${STATE:-0.5479121208190918,-0.12224312126636505,0.7171958684921265,0.39473605155944824,-0.8116453289985657,0.9512447118759155}"

PYTHON="${PYTHON:-python3}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${VLA_CPP_ROOT}/debug/artifacts/robot_server_correctness}"
DTYPE="${DTYPE:-f32}"
# ====================================

LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/smolvla-llm-${DTYPE}.gguf}"
VISION_GGUF="${VISION_GGUF:-${GGUF_DIR}/mmproj-smolvla-${DTYPE}.gguf}"
STATE_PROJ_GGUF="${STATE_PROJ_GGUF:-${GGUF_DIR}/state-proj-smolvla-${DTYPE}.gguf}"
ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF:-${GGUF_DIR}/action-expert-smolvla-${DTYPE}.gguf}"

RAW_PREDICT_BIN="${BUILD_DIR}/bin/smolvla-raw-predict"
LAUNCH_SHELL="${VLA_CPP_ROOT}/robot_server/shell/launch_robot_server_mac_cpu.sh"
CLIENT_SCRIPT="${VLA_CPP_ROOT}/robot_server/test/predict_test_image.py"
RAW_CONVERT_SCRIPT="${VLA_CPP_ROOT}/robot_server/test/smolvla_image_to_raw_rgb.py"
COMPARE_SCRIPT="${VLA_CPP_ROOT}/robot_server/test/compare_smolvla_m7_actions.py"

RAW_DUMP_DIR="${ARTIFACT_DIR}/raw"
SERVER_DUMP_DIR="${ARTIFACT_DIR}/server"
SERVER_LOG="${ARTIFACT_DIR}/server.log"
RAW_RGB_FILE="${ARTIFACT_DIR}/input/raw_rgb_u8.bin"
RAW_META_FILE="${ARTIFACT_DIR}/input/raw_rgb_u8.meta"

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
    -DGGML_METAL=OFF \
    -DVLACPP_BUILD_SMOLVLA=ON \
    -DVLACPP_BUILD_TESTS=ON

echo "== build =="
"${CMAKE_BIN}" --build "${BUILD_DIR}" \
    --target model-server smolvla-raw-predict \
    -j8

echo "== prepare outputs =="
rm -rf "${RAW_DUMP_DIR}" "${SERVER_DUMP_DIR}" "${ARTIFACT_DIR}/input"
mkdir -p "${RAW_DUMP_DIR}" "${SERVER_DUMP_DIR}" "${ARTIFACT_DIR}/input"

echo "== prepare raw rgb =="
"${PYTHON}" "${RAW_CONVERT_SCRIPT}" \
    --image "${IMAGE_PATH}" \
    --raw-out "${RAW_RGB_FILE}" \
    --meta-out "${RAW_META_FILE}"

RAW_WIDTH="$(awk '$1 == "width" { print $2 }' "${RAW_META_FILE}")"
RAW_HEIGHT="$(awk '$1 == "height" { print $2 }' "${RAW_META_FILE}")"
RAW_STRIDE="$(awk '$1 == "stride_bytes" { print $2 }' "${RAW_META_FILE}")"

echo "== run raw reference =="
"${RAW_PREDICT_BIN}" \
    --llm "${LLM_GGUF}" \
    --mmproj "${VISION_GGUF}" \
    --state-proj "${STATE_PROJ_GGUF}" \
    --action-expert "${ACTION_EXPERT_GGUF}" \
    --raw-rgb "${RAW_RGB_FILE}" \
    --width "${RAW_WIDTH}" \
    --height "${RAW_HEIGHT}" \
    --stride-bytes "${RAW_STRIDE}" \
    --state "${STATE}" \
    --task "${PROMPT}" \
    --threads "${THREADS}" \
    --noise-mode debug-sin \
    --noise-seed -1 \
    --dump-dir "${RAW_DUMP_DIR}" \
    --verbosity 0

echo "== launch server =="
VLA_CPP_ROOT="${VLA_CPP_ROOT}" \
BUILD_DIR="${BUILD_DIR}" \
GGUF_DIR="${GGUF_DIR}" \
DTYPE="${DTYPE}" \
LLM_GGUF="${LLM_GGUF}" \
VISION_GGUF="${VISION_GGUF}" \
STATE_PROJ_GGUF="${STATE_PROJ_GGUF}" \
ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF}" \
HOST="${HOST}" \
PORT="${PORT}" \
THREADS="${THREADS}" \
TASK="${PROMPT}" \
NOISE_MODE="debug-sin" \
NOISE_SEED="-1" \
SKIP_BUILD="1" \
bash "${LAUNCH_SHELL}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "== wait server =="
sleep 3

echo "== run client example =="
"${PYTHON}" "${CLIENT_SCRIPT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --raw-rgb "${RAW_RGB_FILE}" \
    --width "${RAW_WIDTH}" \
    --height "${RAW_HEIGHT}" \
    --state "${STATE}" \
    --prompt "${PROMPT}" \
    --dump-dir "${SERVER_DUMP_DIR}"

echo "== compare server vs raw reference =="
"${PYTHON}" "${COMPARE_SCRIPT}" \
    --cpp-dir "${RAW_DUMP_DIR}" \
    --py-dir "${SERVER_DUMP_DIR}" \
    --max-abs-threshold 1e-5 \
    --mean-abs-threshold 1e-6 \
    --cos-threshold 0.999999

echo "== done =="
echo "raw dump: ${RAW_DUMP_DIR}"
echo "server dump: ${SERVER_DUMP_DIR}"
echo "server log: ${SERVER_LOG}"
