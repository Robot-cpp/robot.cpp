#!/usr/bin/env bash
# Ensure pyrealsense2-macosx uses its bundled librealsense dylib (do NOT symlink to brew).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[fix] macOS only; skipped." >&2
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "[error] python not found; activate conda env first (e.g. conda activate lerobot-demo)" >&2
  exit 1
fi

_pkg_dir() {
  "${PY}" -c '
import importlib.util
spec = importlib.util.find_spec("pyrealsense2")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)
print(spec.submodule_search_locations[0])
' 2>/dev/null || true
}

PY_PKG="$(_pkg_dir)"
if [[ -z "${PY_PKG}" || ! -d "${PY_PKG}" ]]; then
  echo "[error] pyrealsense2 not installed." >&2
  echo "  Run: pip install -r ${ROOT}/requirements-macos-realsense.txt" >&2
  exit 1
fi

PY_DYLIBS="${PY_PKG}/.dylibs"
if [[ ! -d "${PY_DYLIBS}" ]]; then
  echo "[error] pyrealsense2 install is incomplete (missing ${PY_DYLIBS})." >&2
  echo "  Common cause: partial install or root-owned files under site-packages/pyrealsense2." >&2
  echo "  Fix:" >&2
  echo "    sudo chown -R \"\$(whoami)\" \"${PY_PKG}\"" >&2
  echo "    pip install --force-reinstall --no-cache-dir -r ${ROOT}/requirements-macos-realsense.txt" >&2
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
  echo "  Try:" >&2
  echo "    sudo chown -R \"\$(whoami)\" \"${PY_PKG}\"" >&2
  echo "    pip install --force-reinstall --no-cache-dir -r ${ROOT}/requirements-macos-realsense.txt" >&2
  exit 1
fi

echo "[fix] Homebrew librealsense is for rs-capture/rs-enumerate only; Python uses bundled dylib."
