#!/usr/bin/env bash
set -euo pipefail

ROBOT_CPP_ROOT="${ROBOT_CPP_ROOT:?ROBOT_CPP_ROOT must be set}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
DATASET_KEY="${DATASET_KEY:?DATASET_KEY must be set}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BF16_DIR="${OUTPUT_DIR%/}.bf16"

for directory in "${OUTPUT_DIR}" "${BF16_DIR}"; do
    if [[ -e "${directory}" || -L "${directory}" ]]; then
        echo "error: output directory already exists: ${directory}" >&2
        exit 1
    fi
done

bash "${ROBOT_CPP_ROOT}/tools/apply_patches.sh"
"${PYTHON_BIN}" "${ROBOT_CPP_ROOT}/tools/hf2gguf/bitvla/convert_bitvla_to_gguf.py" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --dataset "${DATASET_KEY}" \
    --dtype bf16 --action-chunk 8 \
    --output "${BF16_DIR}"

"${PYTHON_BIN}" "${ROBOT_CPP_ROOT}/tools/hf2gguf/bitvla/pack_i2s.py" \
    --source "${BF16_DIR}" --dtype bf16 \
    --output "${OUTPUT_DIR}"

rm -r -- "${BF16_DIR}"
