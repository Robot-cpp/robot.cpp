#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
CATALOG="${ROOT_DIR}/tools/hf2gguf/starvla/checkpoint_catalog.json"

usage() {
    echo "Usage: tools/hf2gguf/starvla/convert.sh VARIANT [OUTPUT_DIR]" >&2
}

if [[ $# -eq 1 && ( "$1" == -h || "$1" == --help ) ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

VARIANT=$1
OUTPUT_DIR="${2:-${ROOT_DIR}/ckpts/starvla/gguf/${VARIANT}}"
[[ -x "${PYTHON}" ]] || { echo "error: missing Python: ${PYTHON}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || {
    echo "error: refusing to overwrite output directory: ${OUTPUT_DIR}" >&2
    exit 2
}

export STARVLA_CONFIG_PYTHON="${PYTHON}"
source "${ROOT_DIR}/tools/hf2gguf/starvla/starvla_variant_config.sh"
load_starvla_variant "${VARIANT}"

download_args=(--variant "${VARIANT}")
[[ "${FRAMEWORK}" == fast ]] && download_args+=(--include-fast-weights)
[[ "${STARVLA_LOCAL_FILES_ONLY:-0}" == 1 ]] && download_args+=(--local-files-only)
"${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/download_starvla.py" "${download_args[@]}"

LLAMA_REV="$("${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["source_revisions"]["llama_cpp"])' \
    "${CATALOG}")"
export LLAMA_ROOT="${LLAMA_ROOT:-${ROOT_DIR}/ckpts/starvla/toolchains/llama.cpp-${LLAMA_REV}}"
if [[ ! -d "${LLAMA_ROOT}" ]]; then
    mkdir -p -- "$(dirname -- "${LLAMA_ROOT}")"
    git -C "${ROOT_DIR}/third_party/llama.cpp" worktree add \
        --detach "${LLAMA_ROOT}" "${LLAMA_REV}"
fi

SOURCE_DIR="${ROOT_DIR}/ckpts/starvla/sources/${CHECKPOINT_DIRECTORY}/${CHECKPOINT_REVISION}"
CHECKPOINT="${SOURCE_DIR}/${CHECKPOINT_RELATIVE_PATH}"
BASE_ASSETS="${ROOT_DIR}/ckpts/starvla/sources/${QWEN_DIRECTORY}/${QWEN_REVISION}"
mkdir -p -- "${ROOT_DIR}/ckpts/starvla/work"
WORK_DIR="$(mktemp -d "${ROOT_DIR}/ckpts/starvla/work/.${VARIANT}.XXXXXX")"
cleanup() { rm -rf -- "${WORK_DIR}"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHON VARIANT SOURCE_DIR CHECKPOINT BASE_ASSETS WORK_DIR OUTPUT_DIR LLAMA_ROOT
if [[ "${FRAMEWORK}" != fast ]]; then
    bash "${ROOT_DIR}/tools/hf2gguf/starvla/convert_starvla_all.sh"
else
    CODEC_REV="$("${PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["shared_assets"]["fast_codec"]["revision"])' \
        "${CATALOG}")"
    "${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/convert_starvla_qwen25_fast.py" \
        --checkpoint "${CHECKPOINT}" \
        --source-dir "${SOURCE_DIR}" \
        --qwen-assets "${BASE_ASSETS}" \
        --fast-codec "${ROOT_DIR}/ckpts/starvla/sources/fast-codec/${CODEC_REV}" \
        --staging-dir "${WORK_DIR}/staging" \
        --output-dir "${OUTPUT_DIR}" \
        --llama-root "${LLAMA_ROOT}" \
        --python "${PYTHON}"
fi

echo "StarVLA ${VARIANT} bundle: ${OUTPUT_DIR}"
