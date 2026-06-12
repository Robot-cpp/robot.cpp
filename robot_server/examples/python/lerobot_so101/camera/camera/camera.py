from __future__ import annotations

from typing import Any

import cv2
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import ColorMode

from .config import OpenCVCameraCropConfig


def center_crop_square_rgb(image: NDArray[Any]) -> NDArray[Any]:
    """Center-crop an RGB image to 1:1 using the shorter side."""
    h, w = image.shape[:2]
    if w == h:
        return image
    side = min(w, h)
    if w > h:
        x0 = (w - side) // 2
        return image[:, x0 : x0 + side].copy()
    y0 = (h - side) // 2
    return image[y0 : y0 + side, :].copy()


def postprocess_frame(image: NDArray[Any], config: OpenCVCameraCropConfig) -> NDArray[Any]:
    processed = image
    if config.resize_width is not None and config.resize_height is not None:
        processed = center_crop_square_rgb(processed)
        processed = cv2.resize(
            processed,
            (config.resize_width, config.resize_height),
            interpolation=cv2.INTER_AREA,
        )
    return processed


class OpenCVCameraCrop(Camera):
    """Wraps upstream OpenCVCamera; center-crops to square then resizes after capture."""

    config_class = OpenCVCameraCropConfig
    name = "opencv_crop"

    def __init__(self, config: OpenCVCameraCropConfig):
        super().__init__(config)
        self.config = config
        self._inner = OpenCVCamera(config.to_opencv_config())

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.index_or_path})"

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    def connect(self, warmup: bool = True) -> None:
        self._inner.connect(warmup=warmup)

    def disconnect(self) -> None:
        self._inner.disconnect()

    def _postprocess(self, frame: NDArray[Any]) -> NDArray[Any]:
        return postprocess_frame(frame, self.config)

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        return self._postprocess(self._inner.read(color_mode=color_mode))

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        return self._postprocess(self._inner.async_read(timeout_ms=timeout_ms))

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        return self._postprocess(self._inner.read_latest(max_age_ms=max_age_ms))

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return OpenCVCamera.find_cameras()
