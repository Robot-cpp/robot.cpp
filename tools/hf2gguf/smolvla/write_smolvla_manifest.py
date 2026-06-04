#!/usr/bin/env python3
"""Write a conversion manifest for SmolVLA GGUF outputs."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


GGUF_NAMES = (
    "mmproj-smolvla-{dtype}.gguf",
    "state-proj-smolvla-{dtype}.gguf",
    "action-expert-smolvla-{dtype}.gguf",
    "smolvla-llm-{dtype}.gguf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--surgery-dir", required=True)
    parser.add_argument("--llama-cpp-root", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--force", required=True)
    parser.add_argument("--skip-surgery", required=True)
    parser.add_argument("--dtype", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "conversion_manifest.json"

    files = []
    for template in GGUF_NAMES:
        path = output_dir / template.format(dtype=args.dtype)
        files.append({
            "path": str(path),
            "bytes": path.stat().st_size if path.exists() else None,
        })

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": args.checkpoint_dir,
        "output_dir": args.output_dir,
        "surgery_dir": args.surgery_dir,
        "llama_cpp_root": args.llama_cpp_root,
        "dtype": args.dtype,
        "force": args.force == "1",
        "skip_surgery": args.skip_surgery == "1",
        "python": args.python_bin,
        "python_version": platform.python_version(),
        "files": files,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
