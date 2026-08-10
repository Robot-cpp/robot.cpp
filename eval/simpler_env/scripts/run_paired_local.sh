#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

REFERENCE_SCRIPT="${REPO_ROOT}/eval/simpler_env/scripts/run_python_reference.sh"
CANDIDATE_SCRIPT="${REPO_ROOT}/eval/simpler_env/scripts/run_model_server.sh"
SIMPLER_PYTHON="${SIMPLER_PYTHON:-${PYTHON:-${REPO_ROOT}/ckpts/simpler_env/.venv/bin/python}}"
REFERENCE_PYTHON="${REFERENCE_PYTHON:-${REPO_ROOT}/ckpts/starvla/.venv-official/bin/python}"
SIMPLER_ENV_ROOT="${SIMPLER_ENV_ROOT:-${REPO_ROOT}/ckpts/simpler_env/source/SimplerEnv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/ckpts/starvla}"
STARVLA_SOURCE="${STARVLA_SOURCE:-${CHECKPOINT_ROOT}/source/starvla}"
NVIDIA_VULKAN_RUNTIME="${NVIDIA_VULKAN_RUNTIME:-${REPO_ROOT}/ckpts/simpler_env/nvidia-535.261.03-runtime}"
NVIDIA_VULKAN_ICD="${NVIDIA_VULKAN_ICD:-${REPO_ROOT}/ckpts/simpler_env/nvidia-535.261.03/nvidia_icd.json}"
CANDIDATE_BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_cuda}"
CANDIDATE_SERVER_BIN="${SERVER_BIN:-${CANDIDATE_BUILD_DIR}/bin/model-server}"
source "${REPO_ROOT}/tools/hf2gguf/starvla/starvla_variant_config.sh"
VARIANT="${VARIANT:-oft}"
load_starvla_variant "${VARIANT}"
REFERENCE_CHECKPOINT="${CHECKPOINT_ROOT}/sources/${CHECKPOINT_DIRECTORY}/${CHECKPOINT_REVISION}/${CHECKPOINT_RELATIVE_PATH}"
REFERENCE_QWEN_ASSETS="${CHECKPOINT_ROOT}/sources/${QWEN_DIRECTORY}/${QWEN_REVISION}"
DEFAULT_REFERENCE_SERVER="${REPO_ROOT}/tools/hf2gguf/starvla/${REFERENCE_SERVER_NAME}"
REFERENCE_SERVER="${REFERENCE_SERVER:-${DEFAULT_REFERENCE_SERVER}}"
CANDIDATE_GGUF_DIR="${GGUF_DIR:-${REPO_ROOT}/ckpts/starvla/gguf/${VARIANT}}"
CANDIDATE_LLM_GGUF="${LLM_GGUF:-${CANDIDATE_GGUF_DIR}/qwen-${ARTIFACT_STEM}-bf16.gguf}"
CANDIDATE_MMPROJ_GGUF="${MMPROJ_GGUF:-${CANDIDATE_GGUF_DIR}/mmproj-${ARTIFACT_STEM}-bf16.gguf}"
CANDIDATE_POLICY_GGUF="${POLICY_GGUF:-${CANDIDATE_GGUF_DIR}/starvla-${ARTIFACT_STEM}-policy-fp32.gguf}"
if [[ "${VARIANT}" == "qwen25_fast" ]]; then
    CANDIDATE_POLICY_GGUF="${POLICY_GGUF:-${CANDIDATE_GGUF_DIR}/policy-qwen25-fast.gguf}"
    CANDIDATE_BUNDLE_MANIFEST="${BUNDLE_MANIFEST:-${CANDIDATE_GGUF_DIR}/qwen25-fast-bundle-manifest.json}"
else
    CANDIDATE_BUNDLE_MANIFEST="${BUNDLE_MANIFEST:-${CANDIDATE_GGUF_DIR}/conversion_manifest.json}"
fi

HOST="${HOST:-127.0.0.1}"
TASK_IDS="${TASK_IDS:-0,1,2,3}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORTS="${PORTS:-5600,5601,5602,5603}"
EPISODE_IDS="${EPISODE_IDS:-0:24}"
REPEATS="${REPEATS:-1}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-1000}"
UNNORM_KEY="${UNNORM_KEY:-${DEFAULT_UNNORM_KEY}}"
COMPARISON_ID="${COMPARISON_ID:-${VARIANT}-local-python-vs-cpp-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/ckpts/starvla/results/${VARIANT}/bridge-local-paired-${COMPARISON_ID}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

parse_list() {
    local raw="$1"
    local -n target="$2"
    raw="${raw//,/ }"
    read -r -a target <<< "${raw}"
}

die() {
    echo "error: $*" >&2
    exit 2
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

[[ "${HOST}" == "127.0.0.1" ]] || die "HOST must be 127.0.0.1"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer"
[[ "${NOISE_SEED_BASE}" =~ ^[0-9]+$ ]] || die "NOISE_SEED_BASE must be a non-negative integer"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${ALLOW_PARTIAL}" == "0" || "${ALLOW_PARTIAL}" == "1" ]] || die "ALLOW_PARTIAL must be 0 or 1"
[[ "${COMPARISON_ID}" =~ ^[A-Za-z0-9._:-]+$ ]] || die "COMPARISON_ID contains unsafe characters"

declare -a tasks=()
declare -a gpus=()
declare -a ports=()
parse_list "${TASK_IDS}" tasks
parse_list "${GPU_IDS}" gpus
parse_list "${PORTS}" ports
(( ${#tasks[@]} == 4 )) || die "TASK_IDS must contain exactly four Bridge task ids"
(( ${#gpus[@]} == ${#tasks[@]} )) || die "GPU_IDS must contain one GPU id per task shard"
(( ${#ports[@]} == ${#tasks[@]} )) || die "PORTS must contain one port per task shard"

declare -A seen_tasks=()
declare -A seen_gpus=()
declare -A seen_ports=()
for index in "${!tasks[@]}"; do
    task="${tasks[index]}"
    gpu="${gpus[index]}"
    port="${ports[index]}"
    [[ "${task}" =~ ^[0-3]$ ]] || die "invalid Bridge task id: ${task}"
    [[ -n "${gpu}" && "${gpu}" != *[[:space:]]* ]] || die "invalid GPU id: ${gpu}"
    [[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) \
        || die "invalid TCP port: ${port}"
    [[ -z "${seen_tasks[${task}]+x}" ]] || die "duplicate task id: ${task}"
    [[ -z "${seen_gpus[${gpu}]+x}" ]] || die "duplicate GPU id: ${gpu}"
    [[ -z "${seen_ports[${port}]+x}" ]] || die "duplicate TCP port: ${port}"
    seen_tasks["${task}"]=1
    seen_gpus["${gpu}"]=1
    seen_ports["${port}"]=1
done
for required_task in 0 1 2 3; do
    [[ -n "${seen_tasks[${required_task}]+x}" ]] || die "TASK_IDS must cover task ${required_task}"
done

for path in \
    "${REFERENCE_SCRIPT}" \
    "${CANDIDATE_SCRIPT}" \
    "${REFERENCE_SERVER}" \
    "${REFERENCE_CHECKPOINT}" \
    "${REFERENCE_QWEN_ASSETS}" \
    "${STARVLA_SOURCE}" \
    "${SIMPLER_ENV_ROOT}" \
    "${NVIDIA_VULKAN_RUNTIME}" \
    "${NVIDIA_VULKAN_ICD}" \
    "${CANDIDATE_LLM_GGUF}" \
    "${CANDIDATE_MMPROJ_GGUF}" \
    "${CANDIDATE_POLICY_GGUF}"; do
    [[ -e "${path}" ]] || die "required paired-eval input was not found: ${path}"
done
[[ -f "${CANDIDATE_BUNDLE_MANIFEST}" ]] \
    || die "candidate conversion manifest was not found: ${CANDIDATE_BUNDLE_MANIFEST}"
[[ ! -e "${REFERENCE_CHECKPOINT}.aria2" ]] \
    || die "reference checkpoint download is incomplete: ${REFERENCE_CHECKPOINT}.aria2"
actual_checkpoint_size="$(stat -c '%s' -- "${REFERENCE_CHECKPOINT}")"
[[ "${actual_checkpoint_size}" == "${CHECKPOINT_SIZE}" ]] \
    || die "reference checkpoint size mismatch: expected ${CHECKPOINT_SIZE}, got ${actual_checkpoint_size}"
for python_bin in "${SIMPLER_PYTHON}" "${REFERENCE_PYTHON}"; do
    [[ -x "${python_bin}" ]] || die "Python interpreter was not found or is not executable: ${python_bin}"
done
command -v setsid >/dev/null 2>&1 || die "setsid is required for process-group cleanup"
command -v realpath >/dev/null 2>&1 || die "realpath is required for output path validation"
[[ -x "${CANDIDATE_SERVER_BIN}" ]] \
    || die "CUDA model-server was not found or is not executable: ${CANDIDATE_SERVER_BIN}"

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"
if [[ "${DRY_RUN}" == "0" ]]; then
    [[ ! -e "${OUTPUT_DIR}" ]] || die "refusing to overwrite existing paired result directory: ${OUTPUT_DIR}"
    mkdir -p -- "$(dirname -- "${OUTPUT_DIR}")"
    mkdir -- "${OUTPUT_DIR}"
    mkdir -- \
        "${OUTPUT_DIR}/reference" \
        "${OUTPUT_DIR}/candidate" \
        "${OUTPUT_DIR}/logs"
fi

vulkan_ld_library_path="${NVIDIA_VULKAN_RUNTIME}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    vulkan_ld_library_path+="${vulkan_ld_library_path:+:}${LD_LIBRARY_PATH}"
fi

declare -a ACTIVE_PGIDS=()

terminate_active_groups() {
    local sent=0
    local pgid
    for pgid in "${ACTIVE_PGIDS[@]}"; do
        if kill -TERM -- "-${pgid}" 2>/dev/null; then
            sent=1
        fi
    done
    if (( sent )); then
        sleep 2
    fi
    for pgid in "${ACTIVE_PGIDS[@]}"; do
        kill -KILL -- "-${pgid}" 2>/dev/null || true
    done
    for pgid in "${ACTIVE_PGIDS[@]}"; do
        wait "${pgid}" 2>/dev/null || true
    done
    ACTIVE_PGIDS=()
}

handle_signal() {
    trap - INT TERM HUP
    echo "paired Bridge evaluation interrupted; terminating all task shards" >&2
    terminate_active_groups
    exit 130
}

trap handle_signal INT TERM HUP
trap terminate_active_groups EXIT

declare -a SHARD_COMMAND=()

build_shard_command() {
    local role="$1"
    local index="$2"
    shift 2
    local task="${tasks[index]}"
    local gpu="${gpus[index]}"
    local port="${ports[index]}"
    local output="${OUTPUT_DIR}/${role}/task-${task}.json"

    SHARD_COMMAND=(
        env
        "PYTHONUNBUFFERED=1"
        "CUDA_VISIBLE_DEVICES=${gpu}"
        "CUDA_DEVICE_ORDER=PCI_BUS_ID"
        "LD_LIBRARY_PATH=${vulkan_ld_library_path}"
        "VK_ICD_FILENAMES=${NVIDIA_VULKAN_ICD}"
        "HOST=${HOST}"
        "PORT=${port}"
        "TASK_IDS=${task}"
        "EPISODE_IDS=${EPISODE_IDS}"
        "REPEATS=${REPEATS}"
        "NOISE_SEED_BASE=${NOISE_SEED_BASE}"
        "COMPARISON_ID=${COMPARISON_ID}"
        "OUTPUT=${output}"
        "SIMPLER_ENV_ROOT=${SIMPLER_ENV_ROOT}"
    )
    if [[ "${role}" == "reference" ]]; then
        SHARD_COMMAND+=(
            "SIMPLER_PYTHON=${SIMPLER_PYTHON}"
            "REFERENCE_PYTHON=${REFERENCE_PYTHON}"
            "VARIANT=${VARIANT}"
            "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
            "STARVLA_SOURCE=${STARVLA_SOURCE}"
            "REFERENCE_SERVER=${REFERENCE_SERVER}"
            "UNNORM_KEY=${UNNORM_KEY}"
            "REFERENCE_METADATA_OUTPUT=${OUTPUT_DIR}/reference/task-${task}.server-metadata.json"
            bash "${REFERENCE_SCRIPT}"
        )
    else
        SHARD_COMMAND+=(
            "PYTHON=${SIMPLER_PYTHON}"
            "VARIANT=${VARIANT}"
            "UNNORM_KEY=${UNNORM_KEY}"
            "BACKEND=linux-cuda"
            "BUILD_DIR=${CANDIDATE_BUILD_DIR}"
            "SERVER_BIN=${CANDIDATE_SERVER_BIN}"
            "GGUF_DIR=${CANDIDATE_GGUF_DIR}"
            "LLM_GGUF=${CANDIDATE_LLM_GGUF}"
            "MMPROJ_GGUF=${CANDIDATE_MMPROJ_GGUF}"
            "POLICY_GGUF=${CANDIDATE_POLICY_GGUF}"
            "BUNDLE_MANIFEST=${CANDIDATE_BUNDLE_MANIFEST}"
            "RESULT_ROLE=candidate_cpp_gguf"
            bash "${CANDIDATE_SCRIPT}"
        )
    fi
    SHARD_COMMAND+=("$@")
}

run_phase() {
    local role="$1"
    shift
    local index
    local log
    local rc
    local remaining
    declare -a phase_pids=()

    echo "${role} phase:"
    for index in "${!tasks[@]}"; do
        build_shard_command "${role}" "${index}" "$@"
        log="${OUTPUT_DIR}/logs/${role}-task-${tasks[index]}.log"
        if [[ "${DRY_RUN}" == "1" ]]; then
            print_command "${SHARD_COMMAND[@]}"
            printf '    stdout/stderr -> %s\n' "${log}"
            continue
        fi
        setsid "${SHARD_COMMAND[@]}" >"${log}" 2>&1 &
        phase_pids+=("$!")
        echo "  task=${tasks[index]} gpu=${gpus[index]} port=${ports[index]} pid=${phase_pids[-1]} log=${log}"
    done
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi

    ACTIVE_PGIDS=("${phase_pids[@]}")
    remaining="${#phase_pids[@]}"
    while (( remaining > 0 )); do
        if wait -n "${phase_pids[@]}"; then
            rc=0
        else
            rc=$?
        fi
        remaining=$((remaining - 1))
        if (( rc != 0 )); then
            echo "${role} phase failed with exit code ${rc}; see ${OUTPUT_DIR}/logs" >&2
            terminate_active_groups
            return "${rc}"
        fi
    done
    ACTIVE_PGIDS=()
    echo "${role} phase complete"
}

run_phase reference "$@"
run_phase candidate "$@"

compare_cmd=("${SIMPLER_PYTHON}" -m eval.simpler_env.runners.compare_local_python)
for task in "${tasks[@]}"; do
    compare_cmd+=(--reference "${OUTPUT_DIR}/reference/task-${task}.json")
done
for task in "${tasks[@]}"; do
    compare_cmd+=(--candidate "${OUTPUT_DIR}/candidate/task-${task}.json")
done
compare_cmd+=(--output "${OUTPUT_DIR}/comparison.json")
[[ "${ALLOW_PARTIAL}" == "1" ]] && compare_cmd+=(--allow-partial)

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "comparison:"
    print_command "${compare_cmd[@]}"
    exit 0
fi

"${compare_cmd[@]}"
echo "paired Bridge comparison: ${OUTPUT_DIR}/comparison.json"
