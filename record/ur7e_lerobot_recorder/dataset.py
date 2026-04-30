from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import CameraSpec
from .runtime import get_lerobot_dataset_cls

DEFAULT_DATASET_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
JOINT_NAMES = [f"joint_{idx}" for idx in range(1, 7)]
TCP_NAMES = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
OBSERVATION_STATE_NAMES = JOINT_NAMES + TCP_NAMES + ["gripper"]
ACTION_NAMES = TCP_NAMES + ["gripper"]
CONVERT_STATE_NAMES = TCP_NAMES + ["gripper"]
CONVERT_ACTION_NAMES = TCP_NAMES + ["gripper"]


@dataclass
class RawSample:
    joints: np.ndarray
    tcp_pose: np.ndarray
    gripper: float
    images: dict[str, np.ndarray]
    skew_ms: float


def build_dataset_features(
    camera_specs: Sequence[CameraSpec],
    *,
    state_names: Sequence[str] = OBSERVATION_STATE_NAMES,
    action_names: Sequence[str] = ACTION_NAMES,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": list(action_names),
        },
    }

    for spec in camera_specs:
        features[spec.feature_key] = {
            "dtype": "video",
            "shape": (spec.height, spec.width, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def _start_dataset_image_writer(dataset, total_threads: int, args) -> None:
    dataset.start_image_writer(
        num_processes=args.image_writer_processes,
        num_threads=total_threads,
    )


def _validate_existing_dataset(dataset, fps: int, features: dict[str, dict[str, Any]]) -> None:
    if dataset.fps != fps:
        raise SystemExit(
            f"已有数据集的 fps 是 {dataset.fps}，但当前命令使用的是 {fps}。"
        )

    expected_feature_keys = set(features) | DEFAULT_DATASET_FEATURE_KEYS
    if set(dataset.features) != expected_feature_keys:
        raise SystemExit(
            "已有数据集里的特征键与当前相机配置不一致。\n"
            "一批 LeRobotDataset 的摄像头数量和名称必须固定。"
        )

    for key, expected_feature in features.items():
        actual_feature = dict(dataset.features.get(key, {}))
        actual_feature.pop("info", None)
        if actual_feature != expected_feature:
            raise SystemExit(
                f"已有数据集中的特征 '{key}' 与当前采集配置不一致。"
            )


def create_or_resume_dataset(
    args,
    camera_specs: Sequence[CameraSpec],
    *,
    state_names: Sequence[str] = OBSERVATION_STATE_NAMES,
    action_names: Sequence[str] = ACTION_NAMES,
):
    LeRobotDataset = get_lerobot_dataset_cls()
    features = build_dataset_features(
        camera_specs,
        state_names=state_names,
        action_names=action_names,
    )
    total_threads = args.image_writer_threads_per_camera * len(camera_specs)

    if args.resume:
        if not args.root.exists():
            raise SystemExit(f"指定了 --resume，但数据集目录不存在: {args.root}")
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.root,
            batch_encoding_size=args.video_encoding_batch_size,
        )
        _validate_existing_dataset(dataset, args.fps, features)
        _start_dataset_image_writer(dataset, total_threads, args)
        return dataset

    if args.root.exists():
        raise SystemExit(
            f"数据集目录已经存在: {args.root}\n"
            "如果你是继续追加数据，请使用 --resume；否则请换一个新的 --root 路径。"
        )

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type=args.robot_type,
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=total_threads,
        batch_encoding_size=args.video_encoding_batch_size,
    )


def observation_state_from_sample(sample: RawSample) -> np.ndarray:
    return np.concatenate(
        [
            sample.joints.astype(np.float32, copy=False),
            sample.tcp_pose.astype(np.float32, copy=False),
            np.asarray([sample.gripper], dtype=np.float32),
        ]
    )


def action_from_sample(sample: RawSample) -> np.ndarray:
    return np.concatenate(
        [
            sample.tcp_pose.astype(np.float32, copy=False),
            np.asarray([sample.gripper], dtype=np.float32),
        ]
    )


def dataset_frame_from_samples(
    previous: RawSample,
    current: RawSample,
    camera_specs: Sequence[CameraSpec],
) -> dict[str, np.ndarray]:
    frame = {
        "observation.state": observation_state_from_sample(previous),
        "action": action_from_sample(current),
    }
    for spec in camera_specs:
        frame[spec.feature_key] = previous.images[spec.name]
    return frame


def validate_existing_dataset(
    dataset,
    fps: int,
    camera_specs: Sequence[CameraSpec],
    *,
    state_names: Sequence[str] = OBSERVATION_STATE_NAMES,
    action_names: Sequence[str] = ACTION_NAMES,
) -> None:
    features = build_dataset_features(
        camera_specs,
        state_names=state_names,
        action_names=action_names,
    )
    _validate_existing_dataset(dataset, fps, features)
