#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "smolvla_common.sh is meant to be sourced" >&2
    exit 2
fi

SMOLVLA_DEBUG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_CPP_ROOT="${VLA_CPP_ROOT:-$(cd "${SMOLVLA_DEBUG_DIR}/.." && pwd)}"
RESEARCH_ROOT="${RESEARCH_ROOT:-$(cd "${VLA_CPP_ROOT}/.." && pwd)}"

BITVLA_ROOT="${BITVLA_ROOT:-${RESEARCH_ROOT}/bitvla.cpp}"
GT_DIR="${GT_DIR:-${BITVLA_ROOT}/3rdparty/bitvla_lerobot/smolvla_cpp_gt}"
GGUF_DIR="${GGUF_DIR:-${RESEARCH_ROOT}/ckpts/smolvla}"

VLM_GGUF="${VLM_GGUF:-${GGUF_DIR}/smolvla-vlm-f32.gguf}"
VISION_GGUF="${VISION_GGUF:-${GGUF_DIR}/mmproj-smolvla-f32.gguf}"
STATE_PROJ_GGUF="${STATE_PROJ_GGUF:-${GGUF_DIR}/state-proj-smolvla-f32.gguf}"
ACTION_EXPERT_GGUF="${ACTION_EXPERT_GGUF:-${GGUF_DIR}/action-expert-smolvla-f32.gguf}"

PRETRAINED_PATH="${PRETRAINED_PATH:-${GGUF_DIR}/pytorch_version/smolvla/smolvla_450m_grasp_50k/pretrained_model}"
SMOLVLA_IMAGE_PATH="${SMOLVLA_IMAGE_PATH:-${GGUF_DIR}/test_image.jpg}"
SMOLVLA_STATE="${SMOLVLA_STATE:-0.5479121208190918,-0.12224312126636505,0.7171958684921265,0.39473605155944824,-0.8116453289985657,0.9512447118759155}"
SMOLVLA_TASK="${SMOLVLA_TASK:-grab the block.}"

DEFAULT_PYTHON_BIN="/opt/homebrew/bin/python3"
if [[ -x "/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

SMOLVLA_BUILD_DIR="${SMOLVLA_BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac}"
SMOLVLA_ARTIFACT_DIR="${SMOLVLA_ARTIFACT_DIR:-${SMOLVLA_DEBUG_DIR}/artifacts/smolvla}"

SMOLVLA_THREADS="${SMOLVLA_THREADS:-8}"
SMOLVLA_SKIP_BUILD="${SMOLVLA_SKIP_BUILD:-0}"

smolvla_die() {
    echo "error: $*" >&2
    exit 1
}

smolvla_require_file() {
    local path="$1"
    local label="$2"
    [[ -f "${path}" ]] || smolvla_die "${label} not found: ${path}"
}

smolvla_require_executable() {
    local path="$1"
    local label="$2"
    [[ -x "${path}" ]] || smolvla_die "${label} not executable: ${path}"
}

smolvla_host_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        echo 8
    fi
}

smolvla_check_common_inputs() {
    smolvla_require_file "${VLM_GGUF}" "VLM GGUF"
    smolvla_require_file "${VISION_GGUF}" "vision GGUF"
    smolvla_require_file "${STATE_PROJ_GGUF}" "state projector GGUF"
    smolvla_require_file "${ACTION_EXPERT_GGUF}" "action expert GGUF"
    smolvla_require_file "${SMOLVLA_IMAGE_PATH}" "test image"
    smolvla_require_executable "${PYTHON_BIN}" "python"
}

smolvla_configure_release_cpu_blas() {
    cmake -S "${VLA_CPP_ROOT}" -B "${SMOLVLA_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=Apple \
        -DGGML_OPENMP=OFF
}
