#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_LLAMA_COMMIT="3e941b813b1acbbf06c2203a94ceb33d84748c1e"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
readonly LLAMA_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/third_party/llama.cpp}"
readonly PATCH_DIR="${REPO_ROOT}/patches/llama.cpp"
readonly PATCHES=(
    "${PATCH_DIR}/0001-qwen3vl-vision-parity.patch"
    "${PATCH_DIR}/0002-per-context-native-graph-control.patch"
)

usage() {
    cat <<'EOF'
Usage: tools/llama_cpp/apply_starvla_patches.sh [--check|--revert]

With no option, apply the StarVLA patches to third_party/llama.cpp.
  --check   Validate the pinned revision and report patch state.
  --revert  Remove an already applied complete patch set.

Set LLAMA_CPP_DIR to validate or patch another checkout of the pinned revision.
EOF
}

mode="apply"
case "${1:-}" in
    "") ;;
    --check) mode="check" ;;
    --revert) mode="revert" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ ! -d "${LLAMA_DIR}/.git" && ! -f "${LLAMA_DIR}/.git" ]]; then
    echo "error: llama.cpp checkout not found: ${LLAMA_DIR}" >&2
    exit 1
fi

actual_commit="$(git -C "${LLAMA_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_LLAMA_COMMIT}" ]]; then
    echo "error: unsupported llama.cpp revision" >&2
    echo "  expected: ${EXPECTED_LLAMA_COMMIT}" >&2
    echo "  actual:   ${actual_commit}" >&2
    exit 1
fi

states=()
for patch in "${PATCHES[@]}"; do
    if git -C "${LLAMA_DIR}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        states+=("applied")
    elif git -C "${LLAMA_DIR}" apply --check "${patch}" >/dev/null 2>&1; then
        states+=("pending")
    else
        echo "error: patch is neither cleanly applicable nor already applied: ${patch}" >&2
        exit 1
    fi
done

all_pending=true
all_applied=true
for state in "${states[@]}"; do
    [[ "${state}" == "pending" ]] || all_pending=false
    [[ "${state}" == "applied" ]] || all_applied=false
done

if [[ "${mode}" == "check" ]]; then
    for i in "${!PATCHES[@]}"; do
        printf '%-8s %s\n' "${states[$i]}" "${PATCHES[$i]#${REPO_ROOT}/}"
    done
    if ! ${all_pending} && ! ${all_applied}; then
        echo "error: partial patch set detected" >&2
        exit 1
    fi
    exit 0
fi

if [[ "${mode}" == "apply" ]]; then
    if ${all_applied}; then
        echo "StarVLA llama.cpp patches are already applied."
        exit 0
    fi
    if ! ${all_pending}; then
        echo "error: refusing to apply a partial patch set" >&2
        exit 1
    fi
    if [[ -n "$(git -C "${LLAMA_DIR}" status --porcelain)" ]]; then
        echo "error: refusing to patch a dirty llama.cpp checkout" >&2
        exit 1
    fi
    for patch in "${PATCHES[@]}"; do
        git -C "${LLAMA_DIR}" apply --check "${patch}"
    done
    for patch in "${PATCHES[@]}"; do
        git -C "${LLAMA_DIR}" apply "${patch}"
        echo "applied ${patch#${REPO_ROOT}/}"
    done
    exit 0
fi

if ${all_pending}; then
    echo "StarVLA llama.cpp patches are not applied."
    exit 0
fi
if ! ${all_applied}; then
    echo "error: refusing to revert a partial patch set" >&2
    exit 1
fi
for ((i=${#PATCHES[@]} - 1; i >= 0; --i)); do
    git -C "${LLAMA_DIR}" apply --reverse --check "${PATCHES[$i]}"
done
for ((i=${#PATCHES[@]} - 1; i >= 0; --i)); do
    git -C "${LLAMA_DIR}" apply --reverse "${PATCHES[$i]}"
    echo "reverted ${PATCHES[$i]#${REPO_ROOT}/}"
done
