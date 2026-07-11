#!/usr/bin/env bash
# Ensure pyrealsense2-macosx uses its bundled librealsense dylib (do NOT symlink to brew).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[fix] macOS only; skipped." >&2
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=so101_env.sh
source "${ROOT}/script/shell/so101_env.sh"

PY="$(_realsense_python)"
PY_DYLIBS="$("${PY}" -c 'import pathlib; print(pathlib.Path(__import__("pyrealsense2").__file__).resolve().parent / ".dylibs")' 2>/dev/null || true)"
if [[ -z "${PY_DYLIBS}" || ! -d "${PY_DYLIBS}" ]]; then
  echo "[error] pyrealsense2 not installed. Run: pip install -r ${ROOT}/requirements-macos-realsense.txt" >&2
  exit 1
fi

BUNDLED="$(ls "${PY_DYLIBS}"/librealsense2.*.dylib 2>/dev/null | head -1 || true)"
if [[ -z "${BUNDLED}" ]]; then
  echo "[error] bundled librealsense dylib not found under ${PY_DYLIBS}" >&2
  exit 1
fi

BACKUP="${BUNDLED}.bak"
if [[ -L "${BUNDLED}" ]]; then
  echo "[fix] Removing brew symlink: ${BUNDLED} -> $(readlink "${BUNDLED}")"
  rm -f "${BUNDLED}"
fi

if [[ ! -f "${BUNDLED}" ]]; then
  if [[ -f "${BACKUP}" ]]; then
    cp -f "${BACKUP}" "${BUNDLED}"
    echo "[fix] Restored bundled dylib from ${BACKUP}"
  else
    echo "[error] missing ${BUNDLED} and no ${BACKUP} backup" >&2
    exit 1
  fi
else
  echo "[fix] Using bundled dylib: ${BUNDLED}"
fi

if ! env -u DYLD_LIBRARY_PATH "${PY}" -c "import pyrealsense2 as rs; print('[fix] pyrealsense2 import ok')" 2>/dev/null; then
  echo "[error] pyrealsense2 still fails to import after restore." >&2
  echo "  Try: pip install --force-reinstall -r ${ROOT}/requirements-macos-realsense.txt" >&2
  exit 1
fi

echo "[fix] Homebrew librealsense is for rs-capture/rs-enumerate only; Python uses bundled dylib."
