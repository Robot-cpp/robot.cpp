#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a test image to RAW_RGB_U8/HWC for SmolVLA server correctness")
    parser.add_argument("--image", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--meta-out", required=True)
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required only for debug/test image conversion") from exc

    with Image.open(args.image) as image:
        image = image.convert("RGB")
        width, height = image.size
        raw = image.tobytes()

    raw_out = Path(args.raw_out)
    meta_out = Path(args.meta_out)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_bytes(raw)
    meta_out.write_text(
        f"width {width}\n"
        f"height {height}\n"
        "channels 3\n"
        f"stride_bytes {width * 3}\n",
        encoding="utf-8",
    )
    print(f"raw_rgb={raw_out} width={width} height={height} channels=3 stride_bytes={width * 3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
