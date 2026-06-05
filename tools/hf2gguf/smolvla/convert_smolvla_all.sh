#!/usr/bin/env bash
set -euo pipefail

# ====== change these if needed ======
VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"                    # vla.cpp root
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"              # SmolVLA safetensors checkpoint
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"                          # candidate GGUF output dir
SURGERY_DIR="${SURGERY_DIR:-${OUTPUT_DIR}/surgery}"                         # intermediate .pt dir

PYTHON_BIN="${PYTHON_BIN:-python3}"                                         # which python you want to use
DTYPE="${DTYPE:?DTYPE must be set}"                                         # f32, f16, or bf16

FORCE="${FORCE:-0}"                                                         # 1 allows overwriting GGUF outputs
SKIP_SURGERY="${SKIP_SURGERY:-0}"                                           # 1 reuses existing surgery outputs
# ====================================

# The tracked entrypoint is rooted by VLA_CPP_ROOT, matching the robot_server
# shell style. Local absolute paths should live in debug/artifacts wrappers.
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${VLA_CPP_ROOT}/third_party/llama.cpp}"   
GGUF_PY_DIR="${LLAMA_CPP_ROOT}/gguf-py"
SMOLVLA_CONVERTER_DIR="${VLA_CPP_ROOT}/tools/hf2gguf/smolvla"

# Put llama.cpp's gguf-py and converter code first so imports use the
# repo-paired GGUF writer and tokenizer pre-tokenizer detection.
export PYTHONPATH="${GGUF_PY_DIR}:${LLAMA_CPP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"


# Do not overwrite candidate/reference GGUF files by accident. Re-running in the
# same OUTPUT_DIR must be explicit via FORCE=1.
for output in \
    "${OUTPUT_DIR}/mmproj-smolvla-${DTYPE}.gguf" \
    "${OUTPUT_DIR}/state-proj-smolvla-${DTYPE}.gguf" \
    "${OUTPUT_DIR}/action-expert-smolvla-${DTYPE}.gguf" \
    "${OUTPUT_DIR}/smolvla-llm-${DTYPE}.gguf"; do
    if [[ -e "${output}" && "${FORCE}" != "1" ]]; then
        echo "error: target already exists: ${output}; set FORCE=1 to overwrite" >&2
        exit 1
    fi
done

echo
echo "== config =="
echo "root:       ${VLA_CPP_ROOT}"
echo "checkpoint: ${CHECKPOINT_DIR}"
echo "output:     ${OUTPUT_DIR}"
echo "surgery:    ${SURGERY_DIR}"
echo "dtype:      ${DTYPE}"
echo "python:     ${PYTHON_BIN}"
echo "converter:  ${SMOLVLA_CONVERTER_DIR}"
echo "gguf-py:    ${GGUF_PY_DIR}"
echo "force:      ${FORCE}"
echo "skip surgery: ${SKIP_SURGERY}"

echo
echo "== surgery =="

cd "${SMOLVLA_CONVERTER_DIR}"
# The surgery step splits the original checkpoint into component .pt files and
# copies config/stat files needed by the component converters.
if [[ "${SKIP_SURGERY}" == "1" ]]; then
    echo "skip surgery: ${SURGERY_DIR}"
else
    echo "run surgery: ${SURGERY_DIR}"
    mkdir -p "${SURGERY_DIR}"
    "${PYTHON_BIN}" smolvla_surgery.py \
        --model "${CHECKPOINT_DIR}" \
        --output-dir "${SURGERY_DIR}"
fi

echo
echo "== convert ${DTYPE} =="



echo "convert vision + connector"
"${PYTHON_BIN}" convert_smolvla_vision_to_gguf.py \
    --surgery-dir "${SURGERY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --dtype "${DTYPE}"

echo "convert state projector"
"${PYTHON_BIN}" convert_smolvla_state_proj_to_gguf.py \
    --surgery-dir "${SURGERY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --dtype "${DTYPE}"

echo "convert action expert"
"${PYTHON_BIN}" convert_smolvla_action_expert_to_gguf.py \
    --surgery-dir "${SURGERY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --dtype "${DTYPE}"

echo "convert LLM backbone"
"${PYTHON_BIN}" convert_smolvla_llm_to_gguf.py \
    --surgery-dir "${SURGERY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --dtype "${DTYPE}"

echo
echo "== manifest =="
# Keep manifest generation in Python so the shell remains a straight pipeline.
"${PYTHON_BIN}" write_smolvla_manifest.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --surgery-dir "${SURGERY_DIR}" \
    --llama-cpp-root "${LLAMA_CPP_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --force "${FORCE}" \
    --skip-surgery "${SKIP_SURGERY}" \
    --dtype "${DTYPE}"

echo
echo "== outputs =="
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "*.gguf" -print | sort
