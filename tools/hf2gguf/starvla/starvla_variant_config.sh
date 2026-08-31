#!/usr/bin/env bash

_STARVLA_CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

load_starvla_variant() {
    if [[ $# -ne 1 ]]; then
        echo "usage: load_starvla_variant VARIANT" >&2
        return 2
    fi

    local config_python="${STARVLA_CONFIG_PYTHON:-python3}"
    local assignments
    assignments="$("${config_python}" - "${_STARVLA_CONFIG_DIR}" "$1" <<'PY'
import shlex
import sys

sys.path.insert(0, sys.argv[1])
from starvla_checkpoint import (  # noqa: E402
    StarVLAError,
    artifact_stem,
    get_qwen_asset,
    get_variant,
    load_catalog,
)

try:
    catalog = load_catalog()
    variant = get_variant(catalog, sys.argv[2])
    _, qwen = get_qwen_asset(catalog, variant)
except StarVLAError as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

checkpoint = variant["checkpoint"]
values = {
    "STARVLA_REVISION": catalog["source_revisions"]["starvla"],
    "MODEL_TYPE": variant["model_type"],
    "FRAMEWORK": variant["framework"],
    "CHECKPOINT_REVISION": variant["revision"],
    "CHECKPOINT_SHA256": checkpoint["sha256"],
    "CHECKPOINT_DIRECTORY": variant["directory"],
    "CHECKPOINT_RELATIVE_PATH": checkpoint["path"],
    "QWEN_REVISION": qwen["revision"],
    "QWEN_DIRECTORY": qwen["directory"],
    "ARTIFACT_STEM": artifact_stem(sys.argv[2]),
}
for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
    )" || return 2
    eval "${assignments}"
}
