#!/usr/bin/env python3
"""Inspect GGUF metadata and tensor directory entries."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


def read_exact(file: Any, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise SystemExit("truncated GGUF file")
    return data


def read_file_string(file: Any) -> str:
    size = struct.unpack("<Q", read_exact(file, 8))[0]
    return read_exact(file, size).decode("utf-8")


def read_file_value(file: Any, value_type: int) -> Any:
    if value_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", read_exact(file, 4))[0]
    if value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", read_exact(file, 4))[0]
    if value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", read_exact(file, 4))[0]
    if value_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<?", read_exact(file, 1))[0])
    if value_type == GGUF_TYPE_STRING:
        return read_file_string(file)
    if value_type == GGUF_TYPE_ARRAY:
        elem_type, count = struct.unpack("<IQ", read_exact(file, 12))
        return [read_file_value(file, elem_type) for _ in range(count)]
    raise SystemExit(f"unsupported GGUF metadata type: {value_type}")


def inspect(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        if read_exact(file, 4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        version, tensor_count, metadata_count = struct.unpack("<IQQ", read_exact(file, 20))
        metadata = {}
        for _ in range(metadata_count):
            key = read_file_string(file)
            value_type = struct.unpack("<I", read_exact(file, 4))[0]
            metadata[key] = read_file_value(file, value_type)

        tensors = []
        for _ in range(tensor_count):
            name = read_file_string(file)
            n_dims = struct.unpack("<I", read_exact(file, 4))[0]
            shape = list(struct.unpack("<" + "Q" * n_dims, read_exact(file, 8 * n_dims)))
            tensor_type, data_offset = struct.unpack("<IQ", read_exact(file, 12))
            tensors.append({"name": name, "shape": shape, "type": tensor_type, "data_offset": data_offset})

    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--contains", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = inspect(args.path)
    tensors = result["tensors"]
    for needle in args.contains:
        tensors = [tensor for tensor in tensors if needle in tensor["name"]]
    result = {**result, "tensor_count": len(tensors), "tensors": tensors}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for tensor in tensors:
            print(f"{tensor['name']}\t{tensor['shape']}\t{tensor['type']}")


if __name__ == "__main__":
    main()
