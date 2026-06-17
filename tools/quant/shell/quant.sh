#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:?must be set}"

export GGUF_DIR="${GGUF_DIR:?GGUF_DIR must be set}"
export QUANT_OUTPUT_DIR="${QUANT_OUTPUT_DIR:?QUANT_OUTPUT_DIR must be set}"

MODEL_QUANT_PYTHON="${MODEL_QUANT_PYTHON:?MODEL_QUANT_PYTHON must be set}"
PYTHON_BIN="${PYTHON_BIN:-${MODEL_QUANT_PYTHON}}"
PLAN_PATH="${PLAN_PATH:-${VLA_ROOT}/tools/quant/config/smolvla_origin.yaml}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
QUANT_BUILD_DIR="${QUANT_BUILD_DIR:-${VLA_ROOT}/build_model_quant}"
CMAKE_BUILD_CONFIG="${CMAKE_BUILD_CONFIG:-Release}"
GGML_BASE_LIB="${GGML_BASE_LIB:-}"

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
  "${CMAKE_BIN}" -S "${VLA_ROOT}" -B "${QUANT_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_CONFIG}" \
    -DVLACPP_BUILD_EXAMPLES=OFF \
    -DVLACPP_BUILD_TESTS=OFF \
    -DVLACPP_BUILD_SMOLVLA=OFF
  "${CMAKE_BIN}" --build "${QUANT_BUILD_DIR}" --target ggml-base --config "${CMAKE_BUILD_CONFIG}" --parallel "$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
  GGML_BASE_LIB="$(find_ggml_base_lib)"
fi

if [[ -z "${GGML_BASE_LIB}" || ! -f "${GGML_BASE_LIB}" ]]; then
  echo "error: failed to locate ggml-base runtime library under ${QUANT_BUILD_DIR}" >&2
  exit 1
fi
export GGML_BASE_LIB

"${PYTHON_BIN}" \
  "${VLA_ROOT}/tools/quant/quantize-vla-gguf.py" \
  "${PLAN_PATH}" \
  --dry-run

"${PYTHON_BIN}" \
  "${VLA_ROOT}/tools/quant/quantize-vla-gguf.py" \
  "${PLAN_PATH}"
