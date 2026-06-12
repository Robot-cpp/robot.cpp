from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@CameraConfig.register_subclass("opencv_crop")
@dataclass
class OpenCVCameraCropConfig(OpenCVCameraConfig):
    """OpenCV camera with center-crop (square) + resize after capture."""

    resize_width: int | None = None
    resize_height: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.resize_width is None) ^ (self.resize_height is None):
            raise ValueError("`resize_width` and `resize_height` must both be set or both be None.")
        if self.resize_width is not None and (self.resize_width <= 0 or self.resize_height <= 0):
            raise ValueError("`resize_width` and `resize_height` must be positive when provided.")

    def image_feature_shape(self) -> tuple[int, int, int]:
        if self.resize_width is not None and self.resize_height is not None:
            return (self.resize_height, self.resize_width, 3)
        return super().image_feature_shape()

    def to_opencv_config(self) -> OpenCVCameraConfig:
        return OpenCVCameraConfig(
            fps=self.fps,
            width=self.width,
            height=self.height,
            index_or_path=self.index_or_path,
            color_mode=self.color_mode,
            rotation=self.rotation,
            warmup_s=self.warmup_s,
            fourcc=self.fourcc,
            backend=self.backend,
        )
