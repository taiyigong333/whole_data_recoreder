#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ur7e_lerobot_recorder.config import CameraSpec
from ur7e_lerobot_recorder.convert import CONVERSION_MANIFEST_NAME, inspect_raw_demos
from ur7e_lerobot_recorder.dataset import (
    CONVERT_ACTION_NAMES,
    CONVERT_STATE_NAMES,
    validate_existing_dataset,
)
from ur7e_lerobot_recorder.runtime import get_lerobot_dataset_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 raw_demos 转换后的 LeRobotDataset v2.1 是否符合预期。"
    )
    parser.add_argument("--root", type=Path, required=True, help="转换后数据集目录。")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="原始 raw_demos 目录。不传时优先从 manifest 中读取 source_dir。",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobotDataset 的 repo_id。不传时优先从 manifest 中读取。",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=("pyav", "video_reader", "torchcodec"),
        help="读取样本时的视频解码后端。当前环境推荐 pyav。",
    )
    parser.add_argument(
        "--num-episode-checks",
        type=int,
        default=3,
        help="抽样检查多少条 episode 的 state/action 数值。默认检查前 3 条。",
    )
    return parser.parse_args()


def _load_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "meta" / CONVERSION_MANIFEST_NAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"[失败] 缺少{label}: {path}")


def _camera_specs_from_manifest(manifest: dict[str, Any], fps: int) -> list[CameraSpec]:
    specs = []
    for camera in manifest.get("camera_specs", []):
        specs.append(
            CameraSpec(
                name=str(camera["name"]),
                backend="offline",
                device=f"manifest:{camera['name']}",
                width=int(camera["width"]),
                height=int(camera["height"]),
                stream_fps=fps,
            )
        )
    return specs


def _np_from_item(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _expected_transition_from_raw(raw_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(raw_path, allow_pickle=True) as data:
        tcp = np.asarray(data["tcp_poses"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32)
        task = str(np.asarray(data["instruction"]).item())
    state = np.concatenate([tcp[0], np.asarray([gripper[0]], dtype=np.float32)]).astype(np.float32)
    action = np.concatenate([tcp[1], np.asarray([gripper[1]], dtype=np.float32)]).astype(np.float32)
    return state, action, task


def _check_structure(root: Path) -> None:
    _require_file(root / "meta" / "info.json", "info.json")
    _require_file(root / "meta" / "episodes.jsonl", "episodes.jsonl")
    _require_file(root / "meta" / "episodes_stats.jsonl", "episodes_stats.jsonl")
    _require_file(root / "meta" / "tasks.jsonl", "tasks.jsonl")
    if not (root / "data").is_dir():
        raise SystemExit(f"[失败] 缺少 data 目录: {root / 'data'}")
    if not (root / "videos").is_dir():
        raise SystemExit(f"[失败] 缺少 videos 目录: {root / 'videos'}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[失败] 数据集目录不存在: {root}")

    _check_structure(root)
    manifest = _load_manifest(root)

    raw_dir = args.raw_dir.expanduser().resolve() if args.raw_dir is not None else None
    if raw_dir is None and manifest is not None and manifest.get("source_dir"):
        raw_dir = Path(manifest["source_dir"]).expanduser().resolve()

    repo_id = args.repo_id
    if repo_id is None and manifest is not None:
        repo_id = manifest.get("repo_id")
    if repo_id is None:
        repo_id = f"local/{root.name}"

    LeRobotDataset = get_lerobot_dataset_cls()
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        video_backend=args.video_backend,
    )

    print("[通过] 数据集目录结构完整")
    print(f"[信息] repo_id={repo_id}")
    print(f"[信息] episodes={dataset.num_episodes}, frames={dataset.num_frames}, fps={dataset.fps}")
    print(f"[信息] camera_keys={dataset.meta.camera_keys}")

    if manifest is not None:
        specs = _camera_specs_from_manifest(manifest, dataset.fps)
        validate_existing_dataset(
            dataset,
            fps=dataset.fps,
            camera_specs=specs,
            state_names=CONVERT_STATE_NAMES,
            action_names=CONVERT_ACTION_NAMES,
        )
        print("[通过] manifest 中的相机分辨率、state/action schema 与数据集一致")

    parquet_count = sum(1 for _ in root.glob("data/chunk-*/*.parquet"))
    video_count = sum(1 for _ in root.glob("videos/chunk-*/*/*.mp4"))
    expected_video_count = dataset.num_episodes * len(dataset.meta.camera_keys)
    if parquet_count != dataset.num_episodes:
        raise SystemExit(
            f"[失败] parquet 文件数不匹配: parquet={parquet_count}, episodes={dataset.num_episodes}"
        )
    if video_count != expected_video_count:
        raise SystemExit(
            f"[失败] mp4 文件数不匹配: mp4={video_count}, expected={expected_video_count}"
        )
    print("[通过] parquet / mp4 文件数与 episode 数量匹配")

    converted_mapping = None
    if manifest is not None:
        converted_mapping = {
            str(name): int(ep_idx)
            for name, ep_idx in manifest.get("converted_episodes", {}).items()
        }
        if len(converted_mapping) != dataset.num_episodes:
            raise SystemExit(
                "[失败] manifest 中 converted_episodes 的数量与数据集 episode 数量不一致: "
                f"manifest={len(converted_mapping)}, dataset={dataset.num_episodes}"
            )
        print("[通过] manifest 中 converted_episodes 数量正确")

    if raw_dir is not None:
        raw_files, raw_config = inspect_raw_demos(raw_dir)
        print(
            f"[信息] raw episodes={len(raw_files)}, raw fps={raw_config.fps}, "
            f"raw main_shape={raw_config.main_shape}, raw wrist_shape={raw_config.wrist_shape}"
        )

        if raw_config.fps != dataset.fps:
            raise SystemExit(
                f"[失败] raw fps 与数据集 fps 不一致: raw={raw_config.fps}, dataset={dataset.fps}"
            )

        if converted_mapping is None:
            converted_names = [path.name for path in raw_files]
        else:
            converted_names = sorted(converted_mapping, key=lambda name: converted_mapping[name])

        expected_frames = 0
        for name in converted_names:
            with np.load(raw_dir / name, allow_pickle=True) as data:
                expected_frames += max(0, int(data["images"].shape[0]) - 1)

        if expected_frames != dataset.num_frames:
            raise SystemExit(
                f"[失败] transition 总数不匹配: expected={expected_frames}, dataset={dataset.num_frames}"
            )
        print("[通过] raw_demos 与转换后数据集的 transition 总数一致")

        checks = converted_names[: max(0, args.num_episode_checks)]
        for file_name in checks:
            ep_idx = converted_mapping[file_name] if converted_mapping is not None else checks.index(file_name)
            global_idx = int(dataset.episode_data_index["from"][ep_idx].item())
            state_expected, action_expected, task_expected = _expected_transition_from_raw(raw_dir / file_name)

            frame = dataset[global_idx]
            state_actual = _np_from_item(frame["observation.state"])
            action_actual = _np_from_item(frame["action"])
            if not np.allclose(state_actual, state_expected, atol=1e-5):
                raise SystemExit(f"[失败] {file_name} 的首条 observation.state 与 raw_demos 不一致")
            if not np.allclose(action_actual, action_expected, atol=1e-5):
                raise SystemExit(f"[失败] {file_name} 的首条 action 与 raw_demos 不一致")
            if frame["task"] != task_expected:
                raise SystemExit(f"[失败] {file_name} 的 task 与 raw_demos 不一致")

        print(f"[通过] 抽样检查了前 {len(checks)} 条 episode 的 state/action/task 数值")

    sample = dataset[0]
    for camera_key in dataset.meta.camera_keys:
        image = _np_from_item(sample[camera_key])
        expected_shape = dataset.features[camera_key]["shape"]
        actual_shape = tuple(int(x) for x in image.shape)
        expected_tensor_shape = (3, expected_shape[0], expected_shape[1])
        if actual_shape != expected_tensor_shape:
            raise SystemExit(
                f"[失败] {camera_key} 样本 shape 不匹配: actual={actual_shape}, expected={expected_tensor_shape}"
            )
    print("[通过] 抽样图像张量 shape 与 meta/info.json 一致")

    print("[完成] 转换后的 LeRobotDataset v2.1 通过检查")


if __name__ == "__main__":
    main()
