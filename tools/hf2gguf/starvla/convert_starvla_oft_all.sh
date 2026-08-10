#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VARIANT=oft
exec "${SCRIPT_DIR}/convert_starvla_all.sh"
