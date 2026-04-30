from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import CameraSpec
from .dataset import (
    CONVERT_ACTION_NAMES,
    CONVERT_STATE_NAMES,
    create_or_resume_dataset,
    validate_existing_dataset,
)
from .runtime import cv2, get_lerobot_dataset_cls, get_video_encoding_manager_cls

CONVERSION_MANIFEST_NAME = "raw_demos_conversion.json"
MAIN_CAMERA_NAME = "cam_high"
WRIST_CAMERA_NAME = "cam_wrist"


@dataclass(frozen=True)
class RawDemoConfig:
    fps: int
    has_wrist: bool
    main_shape: tuple[int, int, int]
    wrist_shape: tuple[int, int, int] | None


def conversion_manifest_path(root: Path) -> Path:
    return root / "meta" / CONVERSION_MANIFEST_NAME


def _resize_frame(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image.astype(np.uint8, copy=False)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False)


def _build_convert_camera_specs(config: RawDemoConfig, args) -> list[CameraSpec]:
    main_width = args.main_width or config.main_shape[1]
    main_height = args.main_height or config.main_shape[0]
    specs = [
        CameraSpec(
            name=MAIN_CAMERA_NAME,
            backend="offline",
            device="raw_demos.images",
            width=main_width,
            height=main_height,
            stream_fps=config.fps,
        )
    ]

    if config.has_wrist:
        assert config.wrist_shape is not None
        wrist_width = args.wrist_width or config.wrist_shape[1]
        wrist_height = args.wrist_height or config.wrist_shape[0]
        specs.append(
            CameraSpec(
                name=WRIST_CAMERA_NAME,
                backend="offline",
                device="raw_demos.images_wrist",
                width=wrist_width,
                height=wrist_height,
                stream_fps=config.fps,
            )
        )

    return specs


def _load_single_raw_demo(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    return payload


def inspect_raw_demos(raw_dir: Path) -> tuple[list[Path], RawDemoConfig]:
    npz_files = sorted(raw_dir.glob("episode_*.npz"))
    if not npz_files:
        raise SystemExit(f"在目录中没有找到任何 episode_*.npz: {raw_dir}")

    fps: int | None = None
    has_wrist: bool | None = None
    main_shape: tuple[int, int, int] | None = None
    wrist_shape: tuple[int, int, int] | None = None

    for npz_path in npz_files:
        payload = _load_single_raw_demo(npz_path)
        if "images" not in payload or "tcp_poses" not in payload or "gripper" not in payload:
            raise SystemExit(f"原始文件缺少必需字段: {npz_path}")
        if "instruction" not in payload or "fps" not in payload:
            raise SystemExit(f"原始文件缺少 instruction/fps 字段: {npz_path}")

        current_fps = int(np.asarray(payload["fps"]).item())
        current_has_wrist = "images_wrist" in payload
        current_main_shape = tuple(int(x) for x in payload["images"].shape[1:])
        current_wrist_shape = (
            tuple(int(x) for x in payload["images_wrist"].shape[1:]) if current_has_wrist else None
        )

        if payload["images"].ndim != 4 or current_main_shape[-1] != 3:
            raise SystemExit(f"`images` 不是合法的 RGB 视频数组: {npz_path}")
        if payload["tcp_poses"].ndim != 2 or payload["tcp_poses"].shape[1] != 6:
            raise SystemExit(f"`tcp_poses` 必须是形状 (T, 6): {npz_path}")
        if payload["gripper"].ndim != 1:
            raise SystemExit(f"`gripper` 必须是一维数组: {npz_path}")
        if payload["images"].shape[0] != payload["tcp_poses"].shape[0] or payload["images"].shape[0] != payload["gripper"].shape[0]:
            raise SystemExit(f"`images`/`tcp_poses`/`gripper` 的长度不一致: {npz_path}")
        if current_has_wrist:
            if payload["images_wrist"].ndim != 4 or current_wrist_shape is None or current_wrist_shape[-1] != 3:
                raise SystemExit(f"`images_wrist` 不是合法的 RGB 视频数组: {npz_path}")
            if payload["images_wrist"].shape[0] != payload["images"].shape[0]:
                raise SystemExit(f"`images_wrist` 与 `images` 的帧数不一致: {npz_path}")

        if fps is None:
            fps = current_fps
        elif fps != current_fps:
            raise SystemExit(
                f"发现不一致的 fps: {npz_path} 是 {current_fps}，但之前的 episode 是 {fps}。"
            )

        if has_wrist is None:
            has_wrist = current_has_wrist
        elif has_wrist != current_has_wrist:
            raise SystemExit(
                "存在部分 episode 带 `images_wrist`、部分不带的情况。"
                "请先整理原始数据，避免单/双相机混用。"
            )

        if main_shape is None:
            main_shape = current_main_shape
        if current_main_shape != main_shape:
            raise SystemExit(
                f"`images` 分辨率不一致: {npz_path} 是 {current_main_shape}，首个 episode 是 {main_shape}。"
            )

        if current_has_wrist:
            if wrist_shape is None:
                wrist_shape = current_wrist_shape
            elif current_wrist_shape != wrist_shape:
                raise SystemExit(
                    f"`images_wrist` 分辨率不一致: {npz_path} 是 {current_wrist_shape}，首个 episode 是 {wrist_shape}。"
                )

    assert fps is not None
    assert has_wrist is not None
    assert main_shape is not None
    return npz_files, RawDemoConfig(
        fps=fps,
        has_wrist=has_wrist,
        main_shape=main_shape,
        wrist_shape=wrist_shape,
    )


def _manifest_payload(
    *,
    args,
    raw_dir: Path,
    config: RawDemoConfig,
    camera_specs: Sequence[CameraSpec],
    converted_episodes: dict[str, int],
) -> dict[str, Any]:
    return {
        "source_dir": str(raw_dir),
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "fps": config.fps,
        "has_wrist": config.has_wrist,
        "state_names": list(CONVERT_STATE_NAMES),
        "action_names": list(CONVERT_ACTION_NAMES),
        "raw_main_shape": list(config.main_shape),
        "raw_wrist_shape": list(config.wrist_shape) if config.wrist_shape is not None else None,
        "camera_specs": [
            {
                "name": spec.name,
                "width": spec.width,
                "height": spec.height,
            }
            for spec in camera_specs
        ],
        "converted_episodes": converted_episodes,
    }


def _load_existing_manifest(root: Path) -> dict[str, Any] | None:
    manifest_path = conversion_manifest_path(root)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    args,
    raw_dir: Path,
    config: RawDemoConfig,
    camera_specs: Sequence[CameraSpec],
) -> dict[str, int]:
    expected = _manifest_payload(
        args=args,
        raw_dir=raw_dir,
        config=config,
        camera_specs=camera_specs,
        converted_episodes=manifest.get("converted_episodes", {}),
    )
    comparable_keys = [
        "source_dir",
        "repo_id",
        "robot_type",
        "fps",
        "has_wrist",
        "state_names",
        "action_names",
        "raw_main_shape",
        "raw_wrist_shape",
        "camera_specs",
    ]
    for key in comparable_keys:
        if manifest.get(key) != expected.get(key):
            raise SystemExit(
                f"现有转换清单中的 `{key}` 与当前参数不一致，拒绝继续追加。\n"
                f"manifest={manifest.get(key)!r}\ncurrent={expected.get(key)!r}"
            )
    converted = manifest.get("converted_episodes", {})
    if not isinstance(converted, dict):
        raise SystemExit("现有 raw_demos_conversion.json 格式损坏：`converted_episodes` 不是对象。")
    return {str(k): int(v) for k, v in converted.items()}


def save_manifest(
    *,
    root: Path,
    args,
    raw_dir: Path,
    config: RawDemoConfig,
    camera_specs: Sequence[CameraSpec],
    converted_episodes: dict[str, int],
) -> None:
    payload = _manifest_payload(
        args=args,
        raw_dir=raw_dir,
        config=config,
        camera_specs=camera_specs,
        converted_episodes=converted_episodes,
    )
    manifest_path = conversion_manifest_path(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_or_create_dataset(args, config: RawDemoConfig, camera_specs: Sequence[CameraSpec]):
    LeRobotDataset = get_lerobot_dataset_cls()
    if args.root.exists():
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.root,
            batch_encoding_size=args.video_encoding_batch_size,
        )
        validate_existing_dataset(
            dataset,
            fps=config.fps,
            camera_specs=camera_specs,
            state_names=CONVERT_STATE_NAMES,
            action_names=CONVERT_ACTION_NAMES,
        )
        total_threads = max(1, args.image_writer_threads) * len(camera_specs)
        dataset.start_image_writer(
            num_processes=args.image_writer_processes,
            num_threads=total_threads,
        )
        return dataset

    create_args = type("ConvertArgs", (), {})()
    create_args.repo_id = args.repo_id
    create_args.root = args.root
    create_args.robot_type = args.robot_type
    create_args.fps = config.fps
    create_args.resume = False
    create_args.image_writer_processes = args.image_writer_processes
    create_args.image_writer_threads_per_camera = max(1, args.image_writer_threads)
    create_args.video_encoding_batch_size = args.video_encoding_batch_size
    return create_or_resume_dataset(
        create_args,
        camera_specs,
        state_names=CONVERT_STATE_NAMES,
        action_names=CONVERT_ACTION_NAMES,
    )


def _frame_from_episode(
    payload: dict[str, Any],
    index: int,
    camera_specs: Sequence[CameraSpec],
) -> dict[str, np.ndarray]:
    tcp_poses = np.asarray(payload["tcp_poses"], dtype=np.float32)
    gripper = np.asarray(payload["gripper"], dtype=np.float32)

    frame = {
        "observation.state": np.concatenate(
            [
                tcp_poses[index].astype(np.float32, copy=False),
                np.asarray([gripper[index]], dtype=np.float32),
            ]
        ),
        "action": np.concatenate(
            [
                tcp_poses[index + 1].astype(np.float32, copy=False),
                np.asarray([gripper[index + 1]], dtype=np.float32),
            ]
        ),
    }

    for spec in camera_specs:
        if spec.name == MAIN_CAMERA_NAME:
            raw_image = payload["images"][index]
        elif spec.name == WRIST_CAMERA_NAME:
            raw_image = payload["images_wrist"][index]
        else:  # pragma: no cover - defensive
            raise ValueError(f"未知相机名: {spec.name}")
        frame[spec.feature_key] = _resize_frame(raw_image, width=spec.width, height=spec.height)

    return frame


def _skip_reason(frame_count: int) -> str | None:
    if frame_count < 2:
        return f"帧数只有 {frame_count}，小于 2，无法形成 transition"
    return None


def run_conversion(args) -> None:
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise SystemExit(f"原始数据目录不存在: {raw_dir}")

    npz_files, config = inspect_raw_demos(raw_dir)
    args.root = args.root.expanduser().resolve()
    camera_specs = _build_convert_camera_specs(config, args)

    existing_manifest = _load_existing_manifest(args.root) if args.root.exists() else None
    if args.root.exists() and existing_manifest is None:
        raise SystemExit(
            "目标输出目录已经存在，但缺少 meta/raw_demos_conversion.json，"
            "无法安全判断哪些 raw_demos 已经被转换。请换一个新的 --root，"
            "或者清理该目录后重新执行。"
        )
    converted_episodes = (
        _validate_manifest(
            existing_manifest,
            args=args,
            raw_dir=raw_dir,
            config=config,
            camera_specs=camera_specs,
        )
        if existing_manifest is not None
        else {}
    )

    dataset = _load_or_create_dataset(args, config, camera_specs)
    skipped_short: list[tuple[str, str]] = []
    saved_count = 0

    print(
        f"开始转换 raw_demos -> LeRobotDataset v2.1\n"
        f"  原始目录: {raw_dir}\n"
        f"  输出目录: {dataset.root}\n"
        f"  fps: {config.fps}\n"
        f"  相机: {[spec.name for spec in camera_specs]}"
    )

    VideoEncodingManager = get_video_encoding_manager_cls()
    try:
        with VideoEncodingManager(dataset):
            for npz_path in npz_files:
                relative_key = npz_path.name
                if relative_key in converted_episodes:
                    print(f"跳过已转换 episode: {relative_key} -> {converted_episodes[relative_key]}")
                    continue

                payload = _load_single_raw_demo(npz_path)
                frame_count = int(payload["images"].shape[0])
                reason = _skip_reason(frame_count)
                if reason is not None:
                    skipped_short.append((relative_key, reason))
                    print(f"跳过 {relative_key}: {reason}")
                    continue

                task = str(np.asarray(payload["instruction"]).item())
                transition_count = frame_count - 1
                for frame_index in range(transition_count):
                    frame = _frame_from_episode(payload, frame_index, camera_specs)
                    dataset.add_frame(frame=frame, task=task)
                dataset.save_episode()

                episode_index = dataset.num_episodes - 1
                converted_episodes[relative_key] = episode_index
                save_manifest(
                    root=dataset.root,
                    args=args,
                    raw_dir=raw_dir,
                    config=config,
                    camera_specs=camera_specs,
                    converted_episodes=converted_episodes,
                )
                saved_count += 1
                print(
                    f"已转换 {relative_key} -> episode {episode_index} "
                    f"({transition_count} transitions)"
                )
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.stop_image_writer()

    print(
        f"转换完成。新增 episodes={saved_count}，当前数据集 episodes={dataset.num_episodes}，"
        f"frames={dataset.num_frames}"
    )
    if skipped_short:
        print("以下原始轨迹因过短被跳过：")
        for name, reason in skipped_short:
            print(f"  - {name}: {reason}")
