#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
LLAMA_ROOT="${LLAMA_ROOT:?set LLAMA_ROOT to an absolute clean checkout of the catalog-pinned llama.cpp revision}"
source "${ROOT_DIR}/tools/hf2gguf/starvla/starvla_variant_config.sh"
VARIANT="${VARIANT:?set VARIANT to oft, groot, pi_v3, qwen25_oft, qwen25_groot, or qwen25_pi}"
load_starvla_variant "${VARIANT}"

if [[ "${FRAMEWORK}" == "fast" ]]; then
    echo "error: FAST uses tools/hf2gguf/starvla/convert_starvla_qwen25_fast.py" >&2
    exit 1
fi

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the pinned ${VARIANT} .pt file}"
SOURCE_DIR="${SOURCE_DIR:?set SOURCE_DIR to the pinned ${VARIANT} source directory}"
BASE_ASSETS="${BASE_ASSETS:?set BASE_ASSETS to the pinned Qwen-VL asset directory}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/ckpts/starvla/work/${VARIANT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ckpts/starvla/gguf/${VARIANT}}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-2G}"
TEXT_DTYPE="${TEXT_DTYPE:-bf16}"
MMPROJ_DTYPE="${MMPROJ_DTYPE:-bf16}"
POLICY_DTYPE="${POLICY_DTYPE:-fp32}"
TEXT_FILENAME="${TEXT_FILENAME:-qwen-${ARTIFACT_STEM}-${TEXT_DTYPE}.gguf}"
MMPROJ_FILENAME="${MMPROJ_FILENAME:-mmproj-${ARTIFACT_STEM}-${MMPROJ_DTYPE}.gguf}"
POLICY_FILENAME="${POLICY_FILENAME:-starvla-${ARTIFACT_STEM}-policy-${POLICY_DTYPE}.gguf}"
MANIFEST_FILENAME="conversion_manifest.json"
TEXT_METADATA_FILENAME="text-metadata.json"
MMPROJ_METADATA_FILENAME="mmproj-metadata.json"

if [[ -e "${WORK_DIR}" ]] &&
   [[ -n "$(find "${WORK_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "error: WORK_DIR must be empty: ${WORK_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

for filename in \
    "${TEXT_FILENAME}" \
    "${MMPROJ_FILENAME}" \
    "${POLICY_FILENAME}" \
    "${TEXT_METADATA_FILENAME}" \
    "${MMPROJ_METADATA_FILENAME}" \
    "${MANIFEST_FILENAME}"; do
    destination="${OUTPUT_DIR}/${filename}"
    if [[ -e "${destination}" || -L "${destination}" ]]; then
        echo "error: refusing to overwrite existing output: ${destination}" >&2
        exit 1
    fi
done

RUN_OUTPUT_DIR=""
declare -a PUBLISHED_FILES=()
SUCCESS=0

cleanup() {
    status=$?
    trap - EXIT
    set +e
    if [[ -n "${RUN_OUTPUT_DIR}" ]]; then
        rm -rf -- "${RUN_OUTPUT_DIR}"
    fi
    if [[ "${SUCCESS}" != 1 ]]; then
        for published in "${PUBLISHED_FILES[@]}"; do
            rm -f -- "${published}"
        done
        rm -rf -- "${WORK_DIR}"
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "${WORK_DIR}"
RUN_OUTPUT_DIR="$(mktemp -d "${OUTPUT_DIR}/.starvla-${ARTIFACT_STEM}.tmp.XXXXXX")"

publish_file() {
    source=$1
    destination=$2
    if [[ ! -f "${source}" || ! -s "${source}" ]]; then
        echo "error: transaction output is missing or empty: ${source}" >&2
        return 1
    fi
    if ! ln -- "${source}" "${destination}"; then
        echo "error: refusing to overwrite existing output: ${destination}" >&2
        return 1
    fi
    PUBLISHED_FILES+=("${destination}")
    rm -f -- "${source}"
}

"${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/inspect_starvla_checkpoint.py" \
    "${CHECKPOINT}" \
    --variant "${VARIANT}" \
    --source-dir "${SOURCE_DIR}" \
    --output "${WORK_DIR}/inspection.json"

"${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/starvla_surgery.py" \
    "${CHECKPOINT}" \
    --variant "${VARIANT}" \
    --source-dir "${SOURCE_DIR}" \
    --base-assets "${BASE_ASSETS}" \
    --output-dir "${WORK_DIR}/staging" \
    --max-shard-size "${MAX_SHARD_SIZE}"

"${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/convert_starvla_qwen_to_gguf.py" \
    --hf-dir "${WORK_DIR}/staging/hf" \
    --surgery-manifest "${WORK_DIR}/staging/surgery_manifest.json" \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --llama-root "${LLAMA_ROOT}" \
    --text-filename "${TEXT_FILENAME}" \
    --mmproj-filename "${MMPROJ_FILENAME}" \
    --text-dtype "${TEXT_DTYPE}" \
    --mmproj-dtype "${MMPROJ_DTYPE}"

"${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/convert_starvla_policy_to_gguf.py" \
    --variant "${VARIANT}" \
    --policy-dir "${WORK_DIR}/staging/policy" \
    --hf-dir "${WORK_DIR}/staging/hf" \
    --surgery-manifest "${WORK_DIR}/staging/surgery_manifest.json" \
    --output "${RUN_OUTPUT_DIR}/${POLICY_FILENAME}" \
    --dtype "${POLICY_DTYPE}" \
    --text-filename "${TEXT_FILENAME}" \
    --mmproj-filename "${MMPROJ_FILENAME}"

"${PYTHON}" "${ROOT_DIR}/tools/hf2gguf/starvla/validate_starvla_bundle.py" \
    --variant "${VARIANT}" \
    --text "${RUN_OUTPUT_DIR}/${TEXT_FILENAME}" \
    --mmproj "${RUN_OUTPUT_DIR}/${MMPROJ_FILENAME}" \
    --policy "${RUN_OUTPUT_DIR}/${POLICY_FILENAME}" \
    --hf-dir "${WORK_DIR}/staging/hf" \
    --policy-dir "${WORK_DIR}/staging/policy" \
    --surgery-manifest "${WORK_DIR}/staging/surgery_manifest.json" \
    --text-dtype "${TEXT_DTYPE}" \
    --mmproj-dtype "${MMPROJ_DTYPE}" \
    --policy-dtype "${POLICY_DTYPE}" \
    --output "${RUN_OUTPUT_DIR}/${MANIFEST_FILENAME}"

publish_file "${RUN_OUTPUT_DIR}/${TEXT_FILENAME}" "${OUTPUT_DIR}/${TEXT_FILENAME}"
publish_file "${RUN_OUTPUT_DIR}/${MMPROJ_FILENAME}" "${OUTPUT_DIR}/${MMPROJ_FILENAME}"
publish_file "${RUN_OUTPUT_DIR}/${POLICY_FILENAME}" "${OUTPUT_DIR}/${POLICY_FILENAME}"
publish_file "${RUN_OUTPUT_DIR}/${TEXT_METADATA_FILENAME}" "${OUTPUT_DIR}/${TEXT_METADATA_FILENAME}"
publish_file "${RUN_OUTPUT_DIR}/${MMPROJ_METADATA_FILENAME}" "${OUTPUT_DIR}/${MMPROJ_METADATA_FILENAME}"
# The manifest is the bundle commit marker and is intentionally published last.
publish_file "${RUN_OUTPUT_DIR}/${MANIFEST_FILENAME}" "${OUTPUT_DIR}/${MANIFEST_FILENAME}"
SUCCESS=1

echo "StarVLA ${VARIANT} bundle written to ${OUTPUT_DIR}"
