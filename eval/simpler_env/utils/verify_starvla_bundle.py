#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def verify_bundle(
    manifest_path: Path,
    *,
    variant: str,
    checkpoint_revision: str,
    checkpoint_sha256: str,
    qwen_revision: str,
    starvla_revision: str,
    components: dict[str, Path],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not a JSON object")
    _require(manifest.get("variant"), variant, "manifest variant")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("manifest source is missing")
    for key, expected in (
        ("revision", checkpoint_revision),
        ("checkpoint_sha256", checkpoint_sha256),
        ("qwen_revision", qwen_revision),
        ("starvla_revision", starvla_revision),
    ):
        _require(source.get(key), expected, f"manifest source.{key}")

    records = manifest.get("components")
    if not isinstance(records, dict):
        raise ValueError("manifest components are missing")
    filename_key = "path" if variant == "qwen25_fast" else "filename"
    for name, path in components.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"manifest component {name} is missing")
        _require(record.get(filename_key), path.name, f"{name} filename")
        _require(record.get("size"), path.stat().st_size, f"{name} size")
        _require(record.get("sha256"), _sha256(path), f"{name} sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a StarVLA Bridge GGUF bundle")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--qwen-revision", required=True)
    parser.add_argument("--starvla-revision", required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        verify_bundle(
            args.manifest,
            variant=args.variant,
            checkpoint_revision=args.checkpoint_revision,
            checkpoint_sha256=args.checkpoint_sha256,
            qwen_revision=args.qwen_revision,
            starvla_revision=args.starvla_revision,
            components={
                "text": args.text,
                "mmproj": args.mmproj,
                "policy": args.policy,
            },
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"verified StarVLA bundle: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
