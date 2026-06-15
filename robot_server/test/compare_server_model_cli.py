#!/usr/bin/env python3
"""Compare model-server predictions with model-cli for the same observation."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robot_client" / "python"))

from model_client import ModelClient  # noqa: E402


ACTION_HEADER_RE = re.compile(r"Predicted Actions \((\d+) steps x (\d+) dims\)")
ACTION_ROW_RE = re.compile(r"Step\s+(\d+):\s+\[(.*)\]")


def parse_state(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _read_ppm_token(fin: BinaryIO) -> bytes:
    token = bytearray()
    while True:
        ch = fin.read(1)
        if not ch:
            return bytes(token)
        if ch == b"#":
            fin.readline()
            continue
        if ch.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(ch)


def load_ppm_rgb(path: Path) -> tuple[bytes, int, int, int] | None:
    with path.open("rb") as fin:
        if fin.read(2) != b"P6":
            return None
        width = int(_read_ppm_token(fin))
        height = int(_read_ppm_token(fin))
        max_value = int(_read_ppm_token(fin))
        if max_value != 255:
            raise ValueError(f"unsupported PPM max value: {max_value}")
        data = fin.read(width * height * 3)
    if len(data) != width * height * 3:
        raise ValueError(f"short PPM data: got {len(data)}, expected {width * height * 3}")
    return data, width, height, width * 3


def load_image_rgb(path: str) -> tuple[bytes, int, int, int]:
    image_path = Path(path)
    ppm = load_ppm_rgb(image_path)
    if ppm is not None:
        return ppm

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for non-PPM image inputs") from exc

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        return image.tobytes(), width, height, width * 3


def model_cli_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.model_cli,
        "--model-type",
        args.model_type,
        "--image",
        args.image[0],
    ]
    for image_path in args.image[1:]:
        cmd += ["--image", image_path]
    for image_name in args.image_name:
        cmd += ["--image-name", image_name]
    cmd += [
        "--state",
        args.state,
        "--task",
        args.task,
        "--threads",
        str(args.threads),
        "--noise-seed",
        str(args.noise_seed),
    ]
    if args.model_type == "smolvla":
        cmd += [
            "--llm",
            args.llm,
            "--mmproj",
            args.mmproj,
            "--state-proj",
            args.state_proj,
            "--action-expert",
            args.action_expert,
            "--n-batch",
            str(args.n_batch),
            "--n-ctx",
            str(args.n_ctx),
            "--noise-mode",
            args.noise_mode,
        ]
    else:
        cmd += [
            "--vit",
            args.vit,
            "--mmproj",
            args.mmproj,
            "--llm",
            args.llm,
            "--tokenizer",
            args.tokenizer,
            "--state-gguf",
            args.state_gguf,
            "--action-decoder",
            args.action_decoder,
        ]
    return cmd


def parse_model_cli_output(text: str) -> tuple[int, int, dict[int, list[float]]]:
    header = ACTION_HEADER_RE.search(text)
    if not header:
        raise ValueError("model-cli output did not contain an action header")
    chunk_size = int(header.group(1))
    action_dim = int(header.group(2))
    rows: dict[int, list[float]] = {}
    for line in text.splitlines():
        match = ACTION_ROW_RE.search(line)
        if not match:
            continue
        row = int(match.group(1))
        values = [float(item.strip()) for item in match.group(2).split(",") if item.strip()]
        if len(values) != action_dim:
            raise ValueError(f"model-cli row {row} has {len(values)} values, expected {action_dim}")
        rows[row] = values
    if not rows:
        raise ValueError("model-cli output did not contain action rows")
    return chunk_size, action_dim, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cli", required=True)
    parser.add_argument("--model-type", choices=["smolvla", "pi0"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--image-name", action="append", default=[])
    parser.add_argument("--state", required=True)
    parser.add_argument("--task", default="grab the block.")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--noise-mode", choices=["gaussian", "debug-sin", "sin"], default="debug-sin")
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--max-abs-threshold", type=float, default=1.0e-3)
    parser.add_argument("--llm")
    parser.add_argument("--mmproj")
    parser.add_argument("--state-proj")
    parser.add_argument("--action-expert")
    parser.add_argument("--vit")
    parser.add_argument("--tokenizer")
    parser.add_argument("--state-gguf")
    parser.add_argument("--action-decoder")
    args = parser.parse_args()

    if args.model_type == "smolvla":
        required = ["llm", "mmproj", "state_proj", "action_expert"]
    else:
        required = ["vit", "mmproj", "llm", "tokenizer", "state_gguf", "action_decoder"]
    missing = [name.replace("_", "-") for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(f"missing required {args.model_type} args: {', '.join(missing)}")

    if not args.image_name:
        if len(args.image) != 1:
            raise SystemExit("multiple --image inputs require one --image-name per image")
        args.image_name = ["image"]
    if len(args.image) != len(args.image_name):
        raise SystemExit(
            f"--image count ({len(args.image)}) must match --image-name count ({len(args.image_name)})"
        )
    images = [load_image_rgb(image_path) for image_path in args.image]
    response = ModelClient(host=args.host, port=args.port).predict(
        {
            "images": [
                {
                    "name": image_name,
                    "rgb_hwc_u8": rgb,
                    "width": width,
                    "height": height,
                    "stride_bytes": stride,
                }
                for image_name, (rgb, width, height, stride) in zip(args.image_name, images)
            ],
            "state": parse_state(args.state),
            "prompt": args.task,
        }
    )

    completed = subprocess.run(
        model_cli_command(args),
        check=True,
        capture_output=True,
        text=True,
    )
    cli_chunk, cli_dim, cli_rows = parse_model_cli_output(completed.stdout)

    if response.chunk_size != cli_chunk or response.action_dim != cli_dim:
        raise SystemExit(
            f"shape mismatch: server={response.chunk_size}x{response.action_dim} "
            f"model-cli={cli_chunk}x{cli_dim}"
        )

    max_abs = 0.0
    max_label = ""
    for row, cli_values in sorted(cli_rows.items()):
        server_values = response.actions[row]
        for col, cli_value in enumerate(cli_values):
            diff = abs(server_values[col] - cli_value)
            if diff > max_abs:
                max_abs = diff
                max_label = f"row={row} col={col} server={server_values[col]:.8f} cli={cli_value:.8f}"

    print(f"shape: {response.chunk_size}x{response.action_dim}")
    print(f"compared_rows: {sorted(cli_rows)}")
    print(f"max_abs: {max_abs:.8f} {max_label}")
    if len(cli_rows) < response.chunk_size:
        print("note: model-cli prints a subset for large chunks, so only printed rows were compared")
    if max_abs > args.max_abs_threshold:
        raise SystemExit(f"FAILED: max_abs {max_abs:.8f} > {args.max_abs_threshold:.8f}")
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
