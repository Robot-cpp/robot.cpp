#!/usr/bin/env bash
set -euo pipefail

# ====== change these if needed ======
ROBOT_CPP_ROOT="${ROBOT_CPP_ROOT:?ROBOT_CPP_ROOT must be set}"                    # robot.cpp root
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"              # pi0 checkpoint dir or safetensors
OUTPUT_PREFIX="${OUTPUT_PREFIX:?OUTPUT_PREFIX must be set}"                 # output prefix, without .vit.gguf suffix

PYTHON_BIN="${PYTHON_BIN:-python3}"                                         # which python you want to use
DTYPE="${DTYPE:-preserve}"                                                  # preserve, fp32, f16, or bf16
FORCE="${FORCE:-0}"                                                         # 1 allows overwriting GGUF outputs
# ====================================

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROBOT_CPP_ROOT}/third_party/llama.cpp}"
GGUF_PY_DIR="${LLAMA_CPP_ROOT}/gguf-py"
PI0_CONVERTER_DIR="${ROBOT_CPP_ROOT}/tools/hf2gguf/pi0"
CHECKPOINT_DIR="$("${PYTHON_BIN}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))' "${CHECKPOINT_DIR}")"
OUTPUT_PREFIX="$("${PYTHON_BIN}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))' "${OUTPUT_PREFIX}")"

# Put llama.cpp's gguf-py and repo tools first so imports use the checked-out
# GGUF writer implementation instead of a site-packages copy.
export PYTHONPATH="${GGUF_PY_DIR}:${PI0_CONVERTER_DIR}:${LLAMA_CPP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for output in \
    "${OUTPUT_PREFIX}.vit.gguf" \
    "${OUTPUT_PREFIX}.mmproj.gguf" \
    "${OUTPUT_PREFIX}.llm.gguf" \
    "${OUTPUT_PREFIX}.tokenizer.gguf" \
    "${OUTPUT_PREFIX}.state.gguf" \
    "${OUTPUT_PREFIX}.action_decoder.gguf"; do
    if [[ -e "${output}" && "${FORCE}" != "1" ]]; then
        echo "error: target already exists: ${output}; set FORCE=1 to overwrite" >&2
        exit 1
    fi
done

echo
echo "== config =="
echo "root:       ${ROBOT_CPP_ROOT}"
echo "checkpoint: ${CHECKPOINT_DIR}"
echo "output:     ${OUTPUT_PREFIX}.*.gguf"
echo "dtype:      ${DTYPE}"
echo "python:     ${PYTHON_BIN}"
echo "converter:  ${PI0_CONVERTER_DIR}"
echo "gguf-py:    ${GGUF_PY_DIR}"
echo "force:      ${FORCE}"

echo
echo "== convert ${DTYPE} =="
cd "${PI0_CONVERTER_DIR}"
"${PYTHON_BIN}" convert_openpi_to_gguf.py \
    --input "${CHECKPOINT_DIR}" \
    --dtype "${DTYPE}" \
    "${OUTPUT_PREFIX}"

echo
echo "== outputs =="
for output in \
    "${OUTPUT_PREFIX}.vit.gguf" \
    "${OUTPUT_PREFIX}.mmproj.gguf" \
    "${OUTPUT_PREFIX}.llm.gguf" \
    "${OUTPUT_PREFIX}.tokenizer.gguf" \
    "${OUTPUT_PREFIX}.state.gguf" \
    "${OUTPUT_PREFIX}.action_decoder.gguf"; do
    ls -lh "${output}"
done
