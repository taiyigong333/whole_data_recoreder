from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraSpec:
    name: str
    backend: str
    device: int | str
    width: int
    height: int
    stream_fps: int | None
    rotation: int = 0
    warmup_s: int = 1
    use_depth: bool = False

    @property
    def feature_key(self) -> str:
        return f"observation.images.{self.name}"

    @property
    def descriptor(self) -> str:
        return f"{self.backend}:{self.device}"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("参数值必须大于 0。")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("参数值必须大于等于 0。")
    return parsed


def normalize_camera_backend(raw_backend: str) -> str:
    normalized = raw_backend.strip().lower()
    if normalized == "realsense":
        normalized = "intelrealsense"
    if normalized not in {"opencv", "intelrealsense"}:
        raise ValueError(f"不支持的相机类型: {raw_backend}")
    return normalized


def normalize_rotation(raw_rotation: int) -> int:
    if raw_rotation == 270:
        raw_rotation = -90
    if raw_rotation not in {0, 90, 180, -90}:
        raise ValueError(
            f"不支持的 rotation={raw_rotation}，仅支持 0、90、180、-90（或 270）。"
        )
    return raw_rotation


def normalize_scan_camera_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    if normalized == "realsense":
        return "intelrealsense"
    if normalized not in {"all", "opencv", "intelrealsense"}:
        raise ValueError(f"不支持的扫描类型: {raw_type}")
    return normalized


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id", type=str, default="local/ur7e_multicam_dataset")
    parser.add_argument(
        "--root",
        type=Path,
        required=False,
        help="数据集根目录，填精确路径，例如：./data/ur7e_pick_red_block",
    )
    parser.add_argument("--robot-type", type=str, default="ur7e")


def _add_record_args(parser: argparse.ArgumentParser) -> None:
    _add_common_dataset_args(parser)
    parser.add_argument("--task", type=str, required=False, help="任务语言描述。")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.88")

    parser.add_argument(
        "--camera-config",
        type=Path,
        help=(
            "相机配置 JSON 文件。可声明 1/2/3 路相机，以及每一路使用 opencv "
            "还是 intelrealsense。"
        ),
    )
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--left-camera", type=int, default=1)
    parser.add_argument("--right-camera", type=int, default=2)
    parser.add_argument("--scan-cameras", action="store_true")
    parser.add_argument(
        "--scan-camera-type",
        choices=("all", "opencv", "realsense", "intelrealsense"),
        default="all",
        help="扫描本机可见的相机后端类型。",
    )
    parser.add_argument("--max-camera-index", type=non_negative_int, default=10)

    parser.add_argument("--fps", type=positive_int, default=15)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="每条轨迹允许采集的原始观测帧数，默认等于 fps * episode-seconds。",
    )
    parser.add_argument(
        "--min-transitions",
        type=positive_int,
        default=30,
        help="保存一条轨迹所需的最少转移样本数量。",
    )
    parser.add_argument("--num-episodes", type=non_negative_int, default=0)

    parser.add_argument(
        "--image-width",
        type=positive_int,
        default=320,
        help="相机默认宽度。camera-config 里不单独指定时使用它。",
    )
    parser.add_argument(
        "--image-height",
        type=positive_int,
        default=240,
        help="相机默认高度。camera-config 里不单独指定时使用它。",
    )
    parser.add_argument(
        "--camera-stream-fps",
        type=positive_int,
        default=None,
        help=(
            "相机硬件流的默认 fps。对 OpenCV 相机默认不强制设置；"
            "对 RealSense 相机默认回退到录制 fps。"
        ),
    )
    parser.add_argument("--camera-buffer-size", type=positive_int, default=1)
    parser.add_argument(
        "--sync-mode",
        choices=("async", "threaded", "sequential"),
        default="async",
        help="多相机取帧方式。async 会复用 lerobot 相机后台线程的最新帧。",
    )

    parser.add_argument(
        "--gripper-source",
        choices=("do", "di", "tool_do", "tool_di", "manual"),
        default="do",
    )
    parser.add_argument("--gripper-index", type=non_negative_int, default=0)
    parser.add_argument(
        "--manual-gripper-initial",
        type=float,
        default=0.0,
        help="人工夹爪标注的初始值：0 表示打开，1 表示闭合。",
    )

    parser.add_argument("--image-writer-threads-per-camera", type=positive_int, default=2)
    parser.add_argument("--image-writer-processes", type=non_negative_int, default=0)
    parser.add_argument("--video-encoding-batch-size", type=positive_int, default=1)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="录制时是否显示实时预览窗口。",
    )


def _add_convert_args(parser: argparse.ArgumentParser) -> None:
    _add_common_dataset_args(parser)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./raw_demos"),
        help="原始 npz 轨迹目录，默认是 ./raw_demos",
    )
    parser.add_argument(
        "--main-width",
        type=positive_int,
        default=None,
        help="主视角输出宽度。不传则保持原始分辨率。",
    )
    parser.add_argument(
        "--main-height",
        type=positive_int,
        default=None,
        help="主视角输出高度。不传则保持原始分辨率。",
    )
    parser.add_argument(
        "--wrist-width",
        type=positive_int,
        default=None,
        help="腕部视角输出宽度。不传则保持原始分辨率。",
    )
    parser.add_argument(
        "--wrist-height",
        type=positive_int,
        default=None,
        help="腕部视角输出高度。不传则保持原始分辨率。",
    )
    parser.add_argument("--image-writer-processes", type=non_negative_int, default=0)
    parser.add_argument("--image-writer-threads", type=positive_int, default=2)
    parser.add_argument("--video-encoding-batch-size", type=positive_int, default=1)


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UR7e 多机位 LeRobotDataset v2.1 录制与 raw_demos 转换工具。"
    )
    subparsers = parser.add_subparsers(dest="command")

    record_parser = subparsers.add_parser(
        "record",
        help="采集 UR7e 多机位数据并写入 LeRobotDataset v2.1",
    )
    _add_record_args(record_parser)

    convert_parser = subparsers.add_parser(
        "convert",
        help="将 XVLA 的 raw_demos/*.npz 转换为 LeRobotDataset v2.1",
    )
    _add_convert_args(convert_parser)
    return parser


def build_legacy_record_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 UR7e 多机位示教数据采集为 LeRobotDataset v2.1。"
    )
    _add_record_args(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in {"record", "convert"}:
        args = build_main_parser().parse_args(argv)
    else:
        args = build_legacy_record_parser().parse_args(argv)
        args.command = "record"
        args.legacy_invocation = True
        return args

    if args.command is None:
        build_main_parser().print_help()
        raise SystemExit(2)

    args.legacy_invocation = False
    return args


def _validate_record_args(args: argparse.Namespace) -> None:
    if not args.scan_cameras:
        if args.root is None:
            raise SystemExit("除非使用 --scan-cameras，否则必须提供 --root。")
        if not args.task:
            raise SystemExit("除非使用 --scan-cameras，否则必须提供 --task。")

    if args.gripper_source == "manual" and not args.preview:
        raise SystemExit("人工夹爪标注模式要求开启预览窗口。")

    if args.max_frames is None:
        args.max_frames = max(2, int(round(args.fps * args.episode_seconds)))

    if args.root is not None:
        args.root = args.root.expanduser().resolve()

    if args.camera_config is not None:
        args.camera_config = args.camera_config.expanduser().resolve()

    args.scan_camera_type = normalize_scan_camera_type(args.scan_camera_type)


def _validate_convert_args(args: argparse.Namespace) -> None:
    if args.root is None:
        raise SystemExit("convert 模式必须提供 --root。")
    args.root = args.root.expanduser().resolve()
    args.raw_dir = args.raw_dir.expanduser().resolve()

    wrist_fields = [args.wrist_width, args.wrist_height]
    if any(value is None for value in wrist_fields) and any(value is not None for value in wrist_fields):
        raise SystemExit("如果要指定腕部相机分辨率，请同时提供 --wrist-width 和 --wrist-height。")

    main_fields = [args.main_width, args.main_height]
    if any(value is None for value in main_fields) and any(value is not None for value in main_fields):
        raise SystemExit("如果要指定主视角分辨率，请同时提供 --main-width 和 --main-height。")


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "record":
        _validate_record_args(args)
        return
    if args.command == "convert":
        _validate_convert_args(args)
        return
    raise SystemExit(f"不支持的命令: {args.command}")


def _camera_device_from_payload(backend: str, payload: dict[str, Any]) -> int | str:
    if "device" in payload:
        raw_device = payload["device"]
    elif backend == "opencv" and "index_or_path" in payload:
        raw_device = payload["index_or_path"]
    elif backend == "intelrealsense" and "serial_number_or_name" in payload:
        raw_device = payload["serial_number_or_name"]
    else:
        raise ValueError(
            f"相机 '{payload.get('name', '<unknown>')}' 缺少 device/index_or_path/serial_number_or_name。"
        )

    if backend == "opencv":
        if isinstance(raw_device, int):
            return raw_device
        if isinstance(raw_device, str) and raw_device.isdigit():
            return int(raw_device)
        return str(raw_device)

    return str(raw_device)


def _stream_fps_for_backend(
    backend: str,
    payload: dict[str, Any],
    default_record_fps: int,
    default_stream_fps: int | None,
) -> int | None:
    if "fps" in payload and payload["fps"] is not None:
        return int(payload["fps"])

    if default_stream_fps is not None:
        return default_stream_fps

    if backend == "intelrealsense":
        return default_record_fps

    return None


def _camera_spec_from_payload(
    payload: dict[str, Any],
    *,
    default_width: int,
    default_height: int,
    default_record_fps: int,
    default_stream_fps: int | None,
) -> CameraSpec:
    if "name" not in payload:
        raise ValueError("camera-config 中的每一路相机都必须提供 name。")

    backend = normalize_camera_backend(str(payload.get("type", "opencv")))
    device = _camera_device_from_payload(backend, payload)
    width = int(payload.get("width", default_width))
    height = int(payload.get("height", default_height))
    stream_fps = _stream_fps_for_backend(
        backend=backend,
        payload=payload,
        default_record_fps=default_record_fps,
        default_stream_fps=default_stream_fps,
    )
    rotation = normalize_rotation(int(payload.get("rotation", 0)))
    warmup_s = int(payload.get("warmup_s", 1))
    use_depth = bool(payload.get("use_depth", False))

    return CameraSpec(
        name=str(payload["name"]),
        backend=backend,
        device=device,
        width=width,
        height=height,
        stream_fps=stream_fps,
        rotation=rotation,
        warmup_s=warmup_s,
        use_depth=use_depth,
    )


def _validate_camera_specs(camera_specs: list[CameraSpec]) -> None:
    if not camera_specs:
        raise SystemExit("至少需要声明一路相机。")

    names = [spec.name for spec in camera_specs]
    if len(names) != len(set(names)):
        raise SystemExit("相机 name 不能重复。")

    descriptors = [(spec.backend, str(spec.device)) for spec in camera_specs]
    if len(descriptors) != len(set(descriptors)):
        raise SystemExit("同一个相机后端 + 设备标识不能重复声明。")


def build_default_camera_specs(args: argparse.Namespace) -> list[CameraSpec]:
    payloads = [
        {"name": "front", "type": "opencv", "device": args.front_camera},
        {"name": "left_arm", "type": "opencv", "device": args.left_camera},
        {"name": "right_arm", "type": "opencv", "device": args.right_camera},
    ]
    camera_specs = [
        _camera_spec_from_payload(
            payload,
            default_width=args.image_width,
            default_height=args.image_height,
            default_record_fps=args.fps,
            default_stream_fps=args.camera_stream_fps,
        )
        for payload in payloads
    ]
    _validate_camera_specs(camera_specs)
    return camera_specs


def load_camera_specs(args: argparse.Namespace) -> list[CameraSpec]:
    if args.camera_config is None:
        return build_default_camera_specs(args)

    if not args.camera_config.exists():
        raise SystemExit(f"相机配置文件不存在: {args.camera_config}")

    with args.camera_config.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    payloads = data.get("cameras")
    if not isinstance(payloads, list):
        raise SystemExit("camera-config 顶层必须包含数组字段 `cameras`。")

    camera_specs = [
        _camera_spec_from_payload(
            payload,
            default_width=args.image_width,
            default_height=args.image_height,
            default_record_fps=args.fps,
            default_stream_fps=args.camera_stream_fps,
        )
        for payload in payloads
    ]
    _validate_camera_specs(camera_specs)
    return camera_specs
