"""Build ROBOT_CAMERAS JSON from so101_env environment variables."""

from __future__ import annotations

import json
import os
import platform
import sys


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    return int(raw)


def build() -> dict:
    key = os.environ.get("CAMERA_KEY", "camera1")
    driver = os.environ.get("CAMERA_DRIVER", "opencv_crop").lower()
    resize_w = _int("CAMERA_RESIZE_WIDTH", 224)
    resize_h = _int("CAMERA_RESIZE_HEIGHT", 224)
    fps = _int("CAMERA_FPS", 30)
    warmup_s = _int("CAMERA_WARMUP_S", 5)

    if driver == "realsense":
        serial = os.environ.get("REALSENSE_SERIAL", "").strip()
        if serial and not serial.startswith("?"):
            entry: dict = {
                "type": "realsense_crop",
                "serial_number_or_name": serial,
                "resize_width": resize_w,
                "resize_height": resize_h,
                "warmup_s": warmup_s,
            }
            auto = os.environ.get("REALSENSE_AUTO_PROFILE", "").strip().lower()
            if platform.system() == "Darwin" and auto in ("", "1", "true", "yes"):
                return {key: entry}
            entry["width"] = _int("CAMERA_WIDTH", 640)
            entry["height"] = _int("CAMERA_HEIGHT", 480)
            entry["fps"] = fps
            return {key: entry}
        print(
            "[warn] REALSENSE_SERIAL not set. OpenCV index may show SMPTE color bars.",
            file=sys.stderr,
        )
        print(
            "       Run: python -m lerobot.scripts.lerobot_find_cameras realsense",
            file=sys.stderr,
        )

    index_raw = os.environ.get("CAMERA_INDEX", "0")
    index_or_path: int | str = int(index_raw) if index_raw.isdigit() else index_raw
    cfg: dict = {
        "type": "opencv_crop",
        "index_or_path": index_or_path,
        "fps": fps,
        "backend": os.environ.get("CAMERA_BACKEND", "DSHOW"),
        "resize_width": resize_w,
        "resize_height": resize_h,
        "warmup_s": warmup_s,
    }
    width = os.environ.get("CAMERA_WIDTH")
    height = os.environ.get("CAMERA_HEIGHT")
    if width and height:
        cfg["width"] = int(width)
        cfg["height"] = int(height)
    return {key: cfg}


def main() -> int:
    print(json.dumps(build(), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
