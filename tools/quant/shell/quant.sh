#!/usr/bin/env bash
set -euo pipefail

ROBOT_CPP_ROOT="${ROBOT_CPP_ROOT:?must be set}" # The root directory of the repository
export SRC_GGUF_DIR="${SRC_GGUF_DIR:?SRC_GGUF_DIR must be set}" # source GGUF directories
export QUANT_OUTPUT_DIR="${QUANT_OUTPUT_DIR:?QUANT_OUTPUT_DIR must be set}" # output directory for quantized models
MODEL_QUANT_PYTHON="${MODEL_QUANT_PYTHON:?MODEL_QUANT_PYTHON must be set}" # Python interpreter path
PLAN_PATH="${PLAN_PATH:-${ROBOT_CPP_ROOT}/tools/quant/config/smolvla_origin.yaml}" # plan yaml path


PYTHON_BIN="${PYTHON_BIN:-${MODEL_QUANT_PYTHON}}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
QUANT_BUILD_DIR="${QUANT_BUILD_DIR:-${ROBOT_CPP_ROOT}/build_model_quant}"
GGML_BASE_LIB="${GGML_BASE_LIB:-}" # path to the ggml-base runtime library (libggml-base)

find_ggml_base_lib() {
  if [[ ! -d "${QUANT_BUILD_DIR}" ]]; then
    return 0
  fi
  case "$(uname -s)" in
    Darwin)
      find "${QUANT_BUILD_DIR}" -type f -name "libggml-base*.dylib" -print -quit
      ;;
    Linux)
      find "${QUANT_BUILD_DIR}" -type f -name "libggml-base*.so*" -print -quit
      ;;
    MINGW*|MSYS*|CYGWIN*)
      find "${QUANT_BUILD_DIR}" -type f \( -name "ggml-base*.dll" -o -name "libggml-base*.dll" \) -print -quit
      ;;
    *)
      find "${QUANT_BUILD_DIR}" -type f \( -name "libggml-base*.so*" -o -name "libggml-base*.dylib" -o -name "ggml-base*.dll" -o -name "libggml-base*.dll" \) -print -quit
      ;;
  esac
}

if [[ -z "${GGML_BASE_LIB}" ]]; then
  GGML_BASE_LIB="$(find_ggml_base_lib)"
fi

if [[ -z "${GGML_BASE_LIB}" || ! -f "${GGML_BASE_LIB}" || "${FORCE_GGML_BASE_BUILD:-0}" == "1" ]]; then
  "${CMAKE_BIN}" -S "${ROBOT_CPP_ROOT}" -B "${QUANT_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DROBOT_CPP_BUILD_ROBOT_SERVER=OFF
  "${CMAKE_BIN}" --build "${QUANT_BUILD_DIR}" --target ggml-base --config Release --parallel "$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
  GGML_BASE_LIB="$(find_ggml_base_lib)"
fi

if [[ -z "${GGML_BASE_LIB}" || ! -f "${GGML_BASE_LIB}" ]]; then
  echo "error: failed to locate ggml-base runtime library under ${QUANT_BUILD_DIR}" >&2
  exit 1
fi
export GGML_BASE_LIB

"${PYTHON_BIN}" \
  "${ROBOT_CPP_ROOT}/tools/quant/quantize-vla-gguf.py" \
  "${PLAN_PATH}" \
  --dry-run

"${PYTHON_BIN}" \
  "${ROBOT_CPP_ROOT}/tools/quant/quantize-vla-gguf.py" \
  "${PLAN_PATH}"
