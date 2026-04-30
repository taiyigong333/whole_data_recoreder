from __future__ import annotations

import importlib.util
import shutil
from typing import Sequence

from .config import CameraSpec

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from rtde_receive import RTDEReceiveInterface
except ImportError as exc:  # pragma: no cover - runtime dependency
    RTDEReceiveInterface = None
    _RTDE_IMPORT_ERROR = exc
else:
    _RTDE_IMPORT_ERROR = None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def pyrealsense2_available() -> bool:
    return module_available("pyrealsense2")


def get_lerobot_dataset_cls():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def get_video_encoding_manager_cls():
    from lerobot.datasets.video_utils import VideoEncodingManager

    return VideoEncodingManager


def get_camera_factory():
    from lerobot.cameras import make_cameras_from_configs

    return make_cameras_from_configs


def get_camera_config_types():
    from lerobot.cameras import ColorMode, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    return OpenCVCameraConfig, RealSenseCameraConfig, ColorMode, Cv2Rotation


def require_runtime_dependencies(
    *,
    camera_specs: Sequence[CameraSpec] | None = None,
    need_cv2: bool = True,
    need_rtde: bool = True,
    need_ffmpeg: bool = True,
) -> None:
    if need_cv2 and cv2 is None:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "当前环境缺少 OpenCV，请先在 lerobot 环境执行 `python -m pip install opencv-python`。"
        ) from _CV2_IMPORT_ERROR

    if need_rtde and RTDEReceiveInterface is None:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "当前环境缺少 ur-rtde，请先在 lerobot 环境安装它。"
        ) from _RTDE_IMPORT_ERROR

    if need_ffmpeg and shutil.which("ffmpeg") is None:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "PATH 中未找到 ffmpeg。LeRobotDataset v2.1 在视频编码阶段需要 ffmpeg。"
        )

    if camera_specs and any(spec.backend == "intelrealsense" for spec in camera_specs):
        if not pyrealsense2_available():  # pragma: no cover - runtime dependency
            raise SystemExit(
                "当前采集配置包含 RealSense 相机，但环境缺少 pyrealsense2。\n"
                "在 macOS arm64 上，PyPI 没有官方 wheel，建议优先尝试：\n"
                "  conda install -n lerobot -c conda-forge pyrealsense2"
            )
