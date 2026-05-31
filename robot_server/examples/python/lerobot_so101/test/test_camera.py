#!/usr/bin/env python3
"""Standalone camera smoke test aligned with lerobot-so101-client capture path.

Uses the same pipeline as ``SOFollower.get_observation`` + ``make_predict_observation``:
  build_camera_config -> make_cameras_from_configs -> connect -> read_latest -> encode
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

_LEROBOT_SO101_ROOT = Path(__file__).resolve().parents[1]
for _path in (_LEROBOT_SO101_ROOT / "src", _LEROBOT_SO101_ROOT / "src" / "lerobot_camera_crop"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import lerobot_camera_crop  # noqa: F401  # register opencv_crop CameraConfig subclass
import numpy as np
from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot_client.bridge.smolvla import make_predict_observation
from lerobot_client.utils.robot import build_camera_config

# Default matches scripts/run_robot_client.sh (camera1 + warmup_s).
DEFAULT_ROBOT_CAMERAS = (
    '{"camera1":{"type":"opencv_crop","index_or_path":0,"width":1280,"height":720,'
    '"fps":30,"backend":"AVFOUNDATION","resize_width":224,"resize_height":224,'
    '"center_crop_square_before_resize":true,"warmup_s":5}}'
)

_CAMERA_TROUBLESHOOTING = """
Camera connect/read failed. Common fixes on macOS:
  1. List devices:  ./test/run_camera_test.sh --list-cameras
  2. Probe indices:  ./test/run_camera_test.sh --probe
  3. Try another index: CAMERA_INDEX=1 ./test/run_camera_test.sh
  4. System Settings -> Privacy & Security -> Camera -> allow Terminal/iTerm/Cursor
  5. Close Zoom/FaceTime/Photo Booth and other apps using the camera
"""


def configure_logging(*, verbose: bool) -> None:
    """Force readable logs even if LeRobot already configured the root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(handler)


def load_robot_cameras_json(*, robot_cameras: str | None, cameras_json: Path | None) -> str:
    if robot_cameras is not None:
        return robot_cameras
    if cameras_json is not None:
        return cameras_json.read_text()
    return DEFAULT_ROBOT_CAMERAS


def apply_camera_index_override(
    camera_configs: dict[str, Any],
    camera_key: str,
    camera_index: int | str | None,
) -> None:
    if camera_index is None:
        return
    if camera_key not in camera_configs:
        raise KeyError(f"camera_key {camera_key!r} not in config")
    parsed_index: int | str = camera_index
    if isinstance(camera_index, str) and camera_index.isdigit():
        parsed_index = int(camera_index)
    camera_configs[camera_key] = dataclasses.replace(
        camera_configs[camera_key],
        index_or_path=parsed_index,
    )


def _default_opencv_backend() -> int:
    if platform.system() == "Darwin":
        return int(Cv2Backends.AVFOUNDATION.value)
    return int(Cv2Backends.ANY.value)


@contextlib.contextmanager
def _suppress_opencv_stderr():
    """Hide noisy OpenCV device-scan messages on stderr."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


def _silence_opencv_logging() -> None:
    import cv2

    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass


def scan_opencv_cameras(*, max_index: int, backend: int | None = None) -> list[dict[str, Any]]:
    """Scan camera indices with the same backend used by the SO101 client config."""
    import cv2

    backend = _default_opencv_backend() if backend is None else backend
    try:
        backend_name = Cv2Backends(backend).name
    except ValueError:
        backend_name = str(backend)
    results: list[dict[str, Any]] = []

    _silence_opencv_logging()
    with _suppress_opencv_stderr():
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            cap.release()

            results.append(
                {
                    "id": idx,
                    "backend": backend_name,
                    "default_width": width,
                    "default_height": height,
                    "default_fps": fps,
                    "read_ok": bool(ret and frame is not None),
                    "read_shape": tuple(frame.shape) if ret and frame is not None else None,
                }
            )

    return results


def list_available_cameras(*, max_index: int) -> int:
    backend_name = Cv2Backends(_default_opencv_backend()).name
    print(f"Scanning OpenCV indices 0..{max_index} (backend={backend_name})", flush=True)
    cameras = scan_opencv_cameras(max_index=max_index)
    if not cameras:
        print(f"WARNING No OpenCV cameras opened in indices 0..{max_index}.", flush=True)
        print(_CAMERA_TROUBLESHOOTING.strip(), flush=True)
        return 1

    readable = [info for info in cameras if info["read_ok"]]
    print(f"Found {len(cameras)} camera(s), {len(readable)} can read a frame:", flush=True)
    for info in cameras:
        status = "OK" if info["read_ok"] else "OPEN but read failed"
        print(
            f"  index={info['id']} backend={info['backend']} "
            f"default={info['default_width']}x{info['default_height']}@{info['default_fps']}fps "
            f"read={status} shape={info['read_shape']}",
            flush=True,
        )

    if readable:
        print(f"Try: CAMERA_INDEX={readable[0]['id']} ./test/run_camera_test.sh", flush=True)
        return 0

    print("WARNING Cameras open but none returned a frame; check macOS camera permission.", flush=True)
    return 1


def probe_camera_indices(
    *,
    robot_cameras: str | None,
    cameras_json: Path | None,
    camera_key: str,
    max_index: int,
) -> int:
    register_third_party_plugins()
    robot_cameras_json = load_robot_cameras_json(
        robot_cameras=robot_cameras,
        cameras_json=cameras_json,
    )
    base_configs = build_camera_config(robot_cameras_json)
    if camera_key not in base_configs:
        keys = ", ".join(sorted(base_configs))
        raise KeyError(f"camera_key {camera_key!r} not in config ({keys})")

    logging.info("Probing indices 0..%d with the same opencv_crop config as the client.", max_index)
    ok_indices: list[int | str] = []
    for idx in range(max_index + 1):
        configs = {camera_key: dataclasses.replace(base_configs[camera_key], index_or_path=idx, warmup_s=3)}
        cam = make_cameras_from_configs(configs)[camera_key]
        try:
            cam.connect()
            frame = cam.read_latest()
            validate_frame(frame, camera_key, configs[camera_key].image_feature_shape()[:2])
            ok_indices.append(idx)
            logging.info("  index=%d OK shape=%s", idx, frame.shape)
        except Exception as exc:
            logging.info("  index=%d FAIL (%s)", idx, exc)
        finally:
            if cam.is_connected:
                cam.disconnect()

    if not ok_indices:
        logging.error("No working camera index found in 0..%d.", max_index)
        logging.error(_CAMERA_TROUBLESHOOTING.strip())
        return 1

    logging.info("Working indices: %s", ", ".join(str(i) for i in ok_indices))
    logging.info("Example: CAMERA_INDEX=%s ./test/run_camera_test.sh", ok_indices[0])
    return 0


def validate_frame(image: np.ndarray, camera_key: str, expected_hw: tuple[int, int]) -> None:
    expected_h, expected_w = expected_hw
    if image.dtype != np.uint8:
        raise ValueError(f"{camera_key}: expected uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{camera_key}: expected HWC RGB with 3 channels, got shape {image.shape}")
    h, w = image.shape[:2]
    if h != expected_h or w != expected_w:
        raise ValueError(f"{camera_key}: expected {expected_h}x{expected_w}, got {h}x{w}")


def validate_observation_payload(
    observation: dict,
    *,
    camera_key: str,
    expected_hw: tuple[int, int],
) -> None:
    expected_h, expected_w = expected_hw
    img = observation["image"]
    expected_bytes = expected_w * expected_h * 3
    rgb = img["rgb_hwc_u8"]
    if len(rgb) != expected_bytes:
        raise ValueError(f"{camera_key}: rgb_hwc_u8 length {len(rgb)} != {expected_bytes}")
    if img["width"] != expected_w or img["height"] != expected_h:
        raise ValueError(
            f"{camera_key}: observation size {img['width']}x{img['height']} != {expected_w}x{expected_h}"
        )
    if img["stride_bytes"] != expected_w * 3:
        raise ValueError(f"{camera_key}: unexpected stride_bytes {img['stride_bytes']}")


def run_camera_test(
    *,
    robot_cameras: str | None,
    cameras_json: Path | None,
    camera_key: str,
    camera_index: int | str | None,
    frames: int,
    interval_s: float,
    save_dir: Path | None,
    preview: bool,
    dummy_state_dim: int,
) -> int:
    import cv2

    register_third_party_plugins()

    if frames <= 0 and not preview:
        frames = 30
        logging.info("frames=0 requires preview; falling back to 30 frames with --no-preview.")

    robot_cameras_json = load_robot_cameras_json(
        robot_cameras=robot_cameras,
        cameras_json=cameras_json,
    )
    camera_configs = build_camera_config(robot_cameras_json)
    apply_camera_index_override(camera_configs, camera_key, camera_index)
    if camera_key not in camera_configs:
        keys = ", ".join(sorted(camera_configs))
        raise KeyError(f"camera_key {camera_key!r} not in config ({keys})")

    cfg = camera_configs[camera_key]
    expected_h, expected_w, expected_c = cfg.image_feature_shape()
    if expected_c != 3:
        raise ValueError(f"{camera_key}: expected 3 channels, got {expected_c}")
    expected_hw = (expected_h, expected_w)

    cameras = make_cameras_from_configs(camera_configs)
    cam = cameras[camera_key]

    logging.info(
        "Connecting %s (%s index_or_path=%s), expected frame shape=%s",
        camera_key,
        cam,
        cfg.index_or_path,
        (expected_h, expected_w, 3),
    )
    try:
        cam.connect()
    except (TimeoutError, ConnectionError, RuntimeError) as exc:
        logging.error("%s", exc)
        logging.error(_CAMERA_TROUBLESHOOTING.strip())
        raise

    dummy_state = [0.0] * dummy_state_dim
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    continuous = frames <= 0
    frame_limit = None if continuous else frames
    window_name = f"lerobot_so101 camera ({camera_key}) [q=quit]"
    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        logging.info("Preview enabled. Press q in the window to quit.")

    captured = 0
    try:
        i = 0
        while frame_limit is None or i < frame_limit:
            t0 = time.perf_counter()
            image = cam.read_latest()
            read_ms = (time.perf_counter() - t0) * 1000.0

            validate_frame(image, camera_key, expected_hw)
            observation = make_predict_observation(image, dummy_state)
            validate_observation_payload(observation, camera_key=camera_key, expected_hw=expected_hw)
            captured += 1

            if not preview or i == 0 or (i + 1) % 30 == 0:
                total_label = "inf" if continuous else str(frame_limit)
                logging.info(
                    "frame=%d/%s shape=%s read_latest_ms=%.1f",
                    i + 1,
                    total_label,
                    image.shape,
                    read_ms,
                )

            if save_dir is not None:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                out_path = save_dir / f"{camera_key}_{i:04d}.png"
                cv2.imwrite(str(out_path), bgr)
                logging.info("saved %s", out_path)

            if preview:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                overlay = bgr.copy()
                cv2.putText(
                    overlay,
                    f"{camera_key} index={cfg.index_or_path} {image.shape[1]}x{image.shape[0]}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("preview quit (q pressed)")
                    break

            i += 1
            if interval_s > 0 and (frame_limit is None or i < frame_limit):
                time.sleep(interval_s)
    finally:
        cam.disconnect()
        if preview:
            cv2.destroyAllWindows()

    logging.info("Camera test passed (%d frames).", captured)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test SO101 cameras with the same capture/encode path as lerobot-so101-client.",
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--robot-cameras",
        default=None,
        help="Camera JSON string (same as lerobot-so101-client --robot-cameras).",
    )
    src_group.add_argument(
        "--cameras-json",
        type=Path,
        default=None,
        help="Camera JSON file (e.g. configs/cameras/front.json).",
    )
    parser.add_argument(
        "--camera-key",
        default="camera1",
        help="Camera dict key (camera1 for run_robot_client.sh, front for front.json).",
    )
    parser.add_argument(
        "--camera-index",
        default=None,
        help="Override index_or_path in the selected camera config (e.g. 0, 1).",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List OpenCV cameras (AVFOUNDATION on macOS) and exit.",
    )
    parser.add_argument(
        "--list-max-index",
        type=int,
        default=2 if platform.system() == "Darwin" else 7,
        help="Highest index to scan with --list-cameras (default: 2 on macOS, else 7).",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Try indices 0..N with the client config and report which can read frames.",
    )
    parser.add_argument(
        "--probe-max-index",
        type=int,
        default=5,
        help="Highest camera index to try with --probe (default: 5).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Frames to capture; 0 = preview until q (default: 0).",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.0,
        help="Sleep between frames (seconds).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save captured RGB frames as PNG.",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live camera window (default: on). Use --no-preview to disable.",
    )
    parser.add_argument(
        "--dummy-state-dim",
        type=int,
        default=6,
        help="Dummy proprio state length for make_predict_observation.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        if args.list_cameras:
            return list_available_cameras(max_index=max(0, args.list_max_index))
        if args.probe:
            return probe_camera_indices(
                robot_cameras=args.robot_cameras,
                cameras_json=args.cameras_json,
                camera_key=args.camera_key,
                max_index=max(0, args.probe_max_index),
            )
        return run_camera_test(
            robot_cameras=args.robot_cameras,
            cameras_json=args.cameras_json,
            camera_key=args.camera_key,
            camera_index=args.camera_index,
            frames=args.frames,
            interval_s=max(0.0, args.interval_s),
            save_dir=args.save_dir,
            preview=args.preview,
            dummy_state_dim=max(1, args.dummy_state_dim),
        )
    except KeyboardInterrupt:
        logging.info("Interrupted.")
        return 130
    except Exception:
        logging.exception("Camera test failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
