#!/usr/bin/env python3
"""Inspect safetensors headers from local files or HF URLs without full download."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


def resolve_repo_file_url(spec: str) -> str:
    if spec.startswith("hf://"):
        base = "https://huggingface.co"
        ref = "main"
        repo_and_file = spec[len("hf://") :]
    elif spec.startswith("ms://"):
        base = "https://modelscope.cn/models"
        ref = "master"
        repo_and_file = spec[len("ms://") :]
    else:
        raise SystemExit("remote repo paths must start with hf:// or ms://")
    parts = repo_and_file.split("/", 2)
    if len(parts) != 3:
        raise SystemExit("remote repo paths must look like hf://owner/repo/path or ms://owner/repo/path")
    owner, repo, filename = parts
    return f"{base}/{owner}/{repo}/resolve/{ref}/{quote(filename)}"


def read_local_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len).decode("utf-8"))


def read_remote_header(url: str) -> dict[str, Any]:
    first = Request(url, headers={"Range": "bytes=0-7"})
    for attempt in range(3):
        try:
            with urlopen(first, timeout=30) as response:
                header_len = struct.unpack("<Q", response.read(8))[0]
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    second = Request(url, headers={"Range": f"bytes=8-{8 + header_len - 1}"})
    for attempt in range(3):
        try:
            with urlopen(second, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def read_header(path_or_url: str) -> dict[str, Any]:
    if path_or_url.startswith("hf://") or path_or_url.startswith("ms://"):
        return read_remote_header(resolve_repo_file_url(path_or_url))
    if path_or_url.startswith("https://") or path_or_url.startswith("http://"):
        return read_remote_header(path_or_url)
    return read_local_header(Path(path_or_url))


def tensor_rows(header: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        rows.append(
            {
                "name": name,
                "dtype": meta.get("dtype"),
                "shape": meta.get("shape"),
                "data_offsets": meta.get("data_offsets"),
            }
        )
    return sorted(rows, key=lambda item: item["name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="local path, https URL, hf://owner/repo/path, or ms://owner/repo/path")
    parser.add_argument("--contains", action="append", default=[], help="only show tensor names containing this text")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--include-metadata", action="store_true", help="include safetensors header metadata in JSON output")
    args = parser.parse_args()

    header = read_header(args.path)
    rows = tensor_rows(header)
    for needle in args.contains:
        rows = [row for row in rows if needle in row["name"]]
    rows = rows[: args.limit]

    if args.json:
        if args.include_metadata:
            print(json.dumps({"metadata": header.get("__metadata__", {}), "tensors": rows}, indent=2))
            return
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print(f"{row['name']}\t{row['dtype']}\t{row['shape']}")


if __name__ == "__main__":
    main()
