from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


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


@CameraConfig.register_subclass("realsense_crop")
@dataclass
class RealSenseCameraCropConfig(CameraConfig):
    """RealSense color stream with center-crop + resize after capture."""

    serial_number_or_name: str
    resize_width: int | None = 224
    resize_height: int | None = 224
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 5

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        if (self.resize_width is None) ^ (self.resize_height is None):
            raise ValueError("`resize_width` and `resize_height` must both be set or both be None.")
        if self.resize_width is not None and (self.resize_width <= 0 or self.resize_height <= 0):
            raise ValueError("`resize_width` and `resize_height` must be positive when provided.")
        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError("For `fps`, `width` and `height`, either all must be set or none.")

    def image_feature_shape(self) -> tuple[int, int, int]:
        if self.resize_width is not None and self.resize_height is not None:
            return (self.resize_height, self.resize_width, 3)
        if self.width is not None and self.height is not None:
            return (self.height, self.width, 3)
        return (480, 640, 3)

    def to_realsense_config(self) -> RealSenseCameraConfig:
        return RealSenseCameraConfig(
            serial_number_or_name=self.serial_number_or_name,
            fps=self.fps,
            width=self.width,
            height=self.height,
            color_mode=self.color_mode,
            use_depth=self.use_depth,
            rotation=self.rotation,
            warmup_s=self.warmup_s,
        )
