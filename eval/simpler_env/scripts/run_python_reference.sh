#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/tools/hf2gguf/starvla/starvla_variant_config.sh"
VARIANT="${VARIANT:-oft}"
load_starvla_variant "${VARIANT}"
EXPECTED_BACKEND=local-python-checkpoint-reference

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/ckpts/starvla}"
CHECKPOINT="${CHECKPOINT_ROOT}/sources/${CHECKPOINT_DIRECTORY}/${CHECKPOINT_REVISION}/${CHECKPOINT_RELATIVE_PATH}"
QWEN_ASSETS="${CHECKPOINT_ROOT}/sources/${QWEN_DIRECTORY}/${QWEN_REVISION}"
DEFAULT_REFERENCE_SERVER="${REPO_ROOT}/tools/hf2gguf/starvla/${REFERENCE_SERVER_NAME}"
STARVLA_SOURCE="${STARVLA_SOURCE:-${CHECKPOINT_ROOT}/source/starvla}"
REFERENCE_SERVER="${REFERENCE_SERVER:-${DEFAULT_REFERENCE_SERVER}}"
REFERENCE_PYTHON="${REFERENCE_PYTHON:-${CHECKPOINT_ROOT}/.venv-official/bin/python}"
SIMPLER_PYTHON="${SIMPLER_PYTHON:-${PYTHON:-${REPO_ROOT}/ckpts/simpler_env/.venv/bin/python}}"
SIMPLER_ENV_ROOT="${SIMPLER_ENV_ROOT:-${REPO_ROOT}/ckpts/simpler_env/source/SimplerEnv}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda:0}"
UNNORM_KEY="${UNNORM_KEY:-${DEFAULT_UNNORM_KEY}}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-1000}"
SERVER_WAIT_S="${SERVER_WAIT_S:-300}"
COMPARISON_ID="${COMPARISON_ID:-}"
OUTPUT="${OUTPUT:-}"

if [[ -z "${COMPARISON_ID}" ]]; then
    echo "COMPARISON_ID is required for a paired local Python reference result" >&2
    exit 2
fi
if [[ "${HOST}" != "127.0.0.1" ]]; then
    echo "HOST must be 127.0.0.1 for the pinned Python reference server" >&2
    exit 2
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "PORT must be an integer in 1..65535" >&2
    exit 2
fi
if [[ ! "${NOISE_SEED_BASE}" =~ ^[0-9]+$ ]]; then
    echo "NOISE_SEED_BASE must be a non-negative integer" >&2
    exit 2
fi
if [[ -z "${OUTPUT}" ]]; then
    OUTPUT="${REPO_ROOT}/ckpts/starvla/results/${VARIANT}/python-reference-${COMPARISON_ID}.json"
fi
if [[ -z "${REFERENCE_METADATA_OUTPUT:-}" ]]; then
    if [[ "${OUTPUT}" == *.json ]]; then
        REFERENCE_METADATA_OUTPUT="${OUTPUT%.json}.server-metadata.json"
    else
        REFERENCE_METADATA_OUTPUT="${OUTPUT}.server-metadata.json"
    fi
fi

for path in \
    "${CHECKPOINT}" \
    "${QWEN_ASSETS}" \
    "${STARVLA_SOURCE}" \
    "${REFERENCE_SERVER}" \
    "${SIMPLER_ENV_ROOT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "required local Python reference input was not found: ${path}" >&2
        exit 2
    fi
done
if [[ -e "${CHECKPOINT}.aria2" ]]; then
    echo "reference checkpoint download is incomplete: ${CHECKPOINT}.aria2" >&2
    exit 2
fi
actual_checkpoint_size="$(stat -c '%s' -- "${CHECKPOINT}")"
if [[ "${actual_checkpoint_size}" != "${CHECKPOINT_SIZE}" ]]; then
    echo "reference checkpoint size mismatch: expected ${CHECKPOINT_SIZE}, got ${actual_checkpoint_size}" >&2
    exit 2
fi
for python_bin in "${REFERENCE_PYTHON}" "${SIMPLER_PYTHON}"; do
    if [[ ! -x "${python_bin}" ]]; then
        echo "required Python interpreter was not found or is not executable: ${python_bin}" >&2
        exit 2
    fi
done
if [[ "${DRY_RUN:-0}" != "1" && -e "${OUTPUT}" ]]; then
    echo "refusing to overwrite existing reference result: ${OUTPUT}" >&2
    exit 2
fi

eval_cmd=(
    "${SIMPLER_PYTHON}" -m eval.simpler_env.runners.run_model_server
    --launch-server
    --host "${HOST}"
    --port "${PORT}"
    --server-wait-s "${SERVER_WAIT_S}"
    --unnorm-key "${UNNORM_KEY}"
    --variant "${VARIANT}"
    --expected-model-type "${MODEL_TYPE}"
    --expected-checkpoint-revision "${CHECKPOINT_REVISION}"
    --expected-checkpoint-sha256 "${CHECKPOINT_SHA256}"
    --expected-qwen-revision "${QWEN_REVISION}"
    --expected-starvla-revision "${STARVLA_REVISION}"
    --expected-framework "${FRAMEWORK}"
    --expected-backend "${EXPECTED_BACKEND}"
    --result-role reference_python_ckpt
    --comparison-id "${COMPARISON_ID}"
    --server-noise-seed-base "${NOISE_SEED_BASE}"
    --simpler-env-root "${SIMPLER_ENV_ROOT}"
    --output "${OUTPUT}"
)
[[ -n "${TASK_IDS:-}" ]] && eval_cmd+=(--task-ids "${TASK_IDS}")
[[ -n "${EPISODE_IDS:-}" ]] && eval_cmd+=(--episode-ids "${EPISODE_IDS}")
[[ -n "${REPEATS:-}" ]] && eval_cmd+=(--repeats "${REPEATS}")
[[ "${RECORD_VIDEO:-0}" == "1" ]] && eval_cmd+=(--record-video)
eval_cmd+=("$@")
eval_cmd+=(
    --server-command
    "${REFERENCE_PYTHON}" -I
    "${REFERENCE_SERVER}"
    --variant "${VARIANT}"
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --starvla-source "${STARVLA_SOURCE}"
    --device "${DEVICE}"
    --host "${HOST}"
    --port "${PORT}"
    --unnorm-key "${UNNORM_KEY}"
    --noise-seed "${NOISE_SEED_BASE}"
    --metadata-output "${REFERENCE_METADATA_OUTPUT}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${eval_cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${eval_cmd[@]}"
