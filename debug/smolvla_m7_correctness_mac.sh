#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/smolvla_common.sh"

SMOLVLA_CPP_BIN_WAS_SET=0
if [[ -n "${SMOLVLA_CPP_BIN+x}" ]]; then
    SMOLVLA_CPP_BIN_WAS_SET=1
fi
SMOLVLA_CPP_BIN="${SMOLVLA_CPP_BIN:-${SMOLVLA_BUILD_DIR}/bin/smolvla-cli}"

SMOLVLA_NOISE_MODE="${SMOLVLA_NOISE_MODE:-debug-sin}"
SMOLVLA_NOISE_SEED="${SMOLVLA_NOISE_SEED:--1}"
SMOLVLA_VISION_BACKEND="${SMOLVLA_VISION_BACKEND:-cpu}"
SMOLVLA_VERBOSITY="${SMOLVLA_VERBOSITY:-0}"
PY_DEVICE="${PY_DEVICE:-cpu}"
PY_DTYPE="${PY_DTYPE:-f32}"

CPP_DUMP_DIR="${CPP_DUMP_DIR:-${SMOLVLA_ARTIFACT_DIR}/m7_cpp_mac}"
PY_DUMP_DIR="${PY_DUMP_DIR:-${SMOLVLA_ARTIFACT_DIR}/m7_py_mac}"

MAX_ABS_THRESHOLD="${MAX_ABS_THRESHOLD:-0.1}"
MEAN_ABS_THRESHOLD="${MEAN_ABS_THRESHOLD:-0.03}"
COS_THRESHOLD="${COS_THRESHOLD:-0.999}"

smolvla_check_common_inputs
smolvla_require_file "${GT_DIR}/gt_sanity_check_generate.py" "Python GT runner"
smolvla_require_file "${SCRIPT_DIR}/compare_smolvla_m7_actions.py" "compare script"

if [[ "${SMOLVLA_SKIP_BUILD}" != "1" && "${SMOLVLA_CPP_BIN_WAS_SET}" != "1" ]]; then
    echo "[1/4] configure+build smolvla-cli"
    smolvla_configure_release_cpu_blas
    cmake --build "${SMOLVLA_BUILD_DIR}" --target smolvla-cli -j"$(smolvla_host_jobs)"
else
    echo "[1/4] skip build"
fi
smolvla_require_executable "${SMOLVLA_CPP_BIN}" "smolvla-cli"

echo "[2/4] run C++ debug-sin and dump M7"
rm -rf "${CPP_DUMP_DIR}" "${PY_DUMP_DIR}"
mkdir -p "${CPP_DUMP_DIR}" "${PY_DUMP_DIR}"

VERBOSE_ARGS=()
if [[ "${SMOLVLA_VERBOSITY}" != "0" ]]; then
    VERBOSE_ARGS+=("-v")
fi

SMOLVLA_VISION_BACKEND="${SMOLVLA_VISION_BACKEND}" \
SMOLVLA_M7_DUMP_DIR="${CPP_DUMP_DIR}" \
"${SMOLVLA_CPP_BIN}" \
    --vlm "${VLM_GGUF}" \
    --mmproj "${VISION_GGUF}" \
    --state-proj "${STATE_PROJ_GGUF}" \
    --action-expert "${ACTION_EXPERT_GGUF}" \
    --image "${SMOLVLA_IMAGE_PATH}" \
    --state "${SMOLVLA_STATE}" \
    --task "${SMOLVLA_TASK}" \
    --threads "${SMOLVLA_THREADS}" \
    --noise-mode "${SMOLVLA_NOISE_MODE}" \
    --noise-seed "${SMOLVLA_NOISE_SEED}" \
    ${VERBOSE_ARGS[@]+"${VERBOSE_ARGS[@]}"}

echo "[3/4] run Python GT and dump M7"
SMOLVLA_M7_DUMP_DIR="${PY_DUMP_DIR}" \
"${PYTHON_BIN}" "${GT_DIR}/gt_sanity_check_generate.py" \
    --pretrained-path "${PRETRAINED_PATH}" \
    --device "${PY_DEVICE}" \
    --image-path "${SMOLVLA_IMAGE_PATH}" \
    --task "${SMOLVLA_TASK}" \
    --dtype "${PY_DTYPE}"

echo "[4/4] compare final actions"
"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_smolvla_m7_actions.py" \
    --cpp-dir "${CPP_DUMP_DIR}" \
    --py-dir "${PY_DUMP_DIR}" \
    --max-abs-threshold "${MAX_ABS_THRESHOLD}" \
    --mean-abs-threshold "${MEAN_ABS_THRESHOLD}" \
    --cos-threshold "${COS_THRESHOLD}"
