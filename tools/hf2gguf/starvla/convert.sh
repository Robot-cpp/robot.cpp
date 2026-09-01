#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
CATALOG="${ROOT_DIR}/tools/hf2gguf/starvla/checkpoint_catalog.json"

usage() {
    cat >&2 <<'EOF'
Usage: tools/hf2gguf/starvla/convert.sh VARIANT [OUTPUT_DIR] [OPTIONS]

Options:
  --checkpoint PATH  Convert a training checkpoint instead of the catalog release
  --source-dir DIR   Run directory containing config.yaml and dataset_statistics.json
  --unnorm-key KEY   Default normalization profile for a training checkpoint
EOF
}

if [[ $# -eq 1 && ( "$1" == -h || "$1" == --help ) ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

VARIANT=$1
shift
OUTPUT_DIR="${ROOT_DIR}/ckpts/starvla/gguf/${VARIANT}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    OUTPUT_DIR=$1
    shift
fi
CHECKPOINT_OVERRIDE=""
SOURCE_DIR_OVERRIDE=""
UNNORM_KEY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|--source-dir|--unnorm-key)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --checkpoint) CHECKPOINT_OVERRIDE=$2 ;;
                --source-dir) SOURCE_DIR_OVERRIDE=$2 ;;
                --unnorm-key) UNNORM_KEY=$2 ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done
if [[ -z "${CHECKPOINT_OVERRIDE}" && ( -n "${SOURCE_DIR_OVERRIDE}" || -n "${UNNORM_KEY}" ) ]]; then
    echo "error: --source-dir and --unnorm-key require --checkpoint" >&2
    exit 2
fi
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
[[ -n "${CHECKPOINT_OVERRIDE}" ]] && download_args+=(--skip-checkpoint)
[[ "${STARVLA_LOCAL_FILES_ONLY:-0}" == 1 ]] && download_args+=(--local-files-only)
"${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/download_starvla.py" \
    --catalog "${CATALOG}" "${download_args[@]}"

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

if [[ -n "${CHECKPOINT_OVERRIDE}" ]]; then
    CHECKPOINT="$(realpath -- "${CHECKPOINT_OVERRIDE}")"
    if [[ -n "${SOURCE_DIR_OVERRIDE}" ]]; then
        SOURCE_DIR="$(realpath -- "${SOURCE_DIR_OVERRIDE}")"
    else
        checkpoint_dir="$(dirname -- "${CHECKPOINT}")"
        for candidate in "${checkpoint_dir}" "$(dirname -- "${checkpoint_dir}")"; do
            if [[ -f "${candidate}/config.yaml" && -f "${candidate}/dataset_statistics.json" ]]; then
                SOURCE_DIR="${candidate}"
                break
            fi
        done
    fi
    [[ -f "${SOURCE_DIR}/config.yaml" && -f "${SOURCE_DIR}/dataset_statistics.json" ]] || {
        echo "error: cannot find config.yaml and dataset_statistics.json; pass --source-dir" >&2
        exit 2
    }
    LOCAL_CATALOG="${WORK_DIR}/checkpoint_catalog.json"
    "${PYTHON}" - "${CATALOG}" "${LOCAL_CATALOG}" "${VARIANT}" \
        "${CHECKPOINT}" "${SOURCE_DIR}" "${UNNORM_KEY}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).parent))
from starvla_checkpoint import atomic_write_json, load_catalog, local_checkpoint_catalog

catalog = local_checkpoint_catalog(
    load_catalog(Path(sys.argv[1])),
    sys.argv[3],
    Path(sys.argv[4]),
    Path(sys.argv[5]),
    sys.argv[6] or None,
)
atomic_write_json(Path(sys.argv[2]), catalog)
PY
    CATALOG="${LOCAL_CATALOG}"
    export STARVLA_CATALOG="${CATALOG}"
    load_starvla_variant "${VARIANT}"
fi

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
        --catalog "${CATALOG}" \
        --llama-root "${LLAMA_ROOT}" \
        --python "${PYTHON}"
fi

echo "StarVLA ${VARIANT} bundle: ${OUTPUT_DIR}"
