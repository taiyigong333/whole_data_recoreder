#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ur7e_lerobot_recorder.config import CameraSpec
from ur7e_lerobot_recorder.convert import CONVERSION_MANIFEST_NAME, inspect_raw_demos
from ur7e_lerobot_recorder.dataset import (
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


def _feature_names(feature: Any) -> list[str]:
    if isinstance(feature, dict):
        names = feature.get("names")
        if isinstance(names, list):
            return [str(name) for name in names]
    return []


def _schema_names(
    manifest: dict[str, Any] | None,
    dataset,
    *,
    feature_key: str,
    manifest_key: str,
) -> list[str]:
    if manifest is not None:
        manifest_names = manifest.get(manifest_key)
        if isinstance(manifest_names, list) and manifest_names:
            return [str(name) for name in manifest_names]
    return _feature_names(dataset.features.get(feature_key, {}))


def _expected_transition_from_raw(raw_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(raw_path, allow_pickle=True) as data:
        tcp = np.asarray(data["tcp_poses"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32)
        task = str(np.asarray(data["instruction"]).item())
    state = np.concatenate(
        [tcp[0], np.asarray([gripper[0]], dtype=np.float32)]
    ).astype(np.float32)
    action = np.concatenate([tcp[1], np.asarray([gripper[1]], dtype=np.float32)]).astype(np.float32)
    return state, action, task


def _gripper_index(names: list[str]) -> int | None:
    for index, name in enumerate(names):
        if str(name).lower() == "gripper":
            return index
    return None


def _position_indices(names: list[str]) -> list[int]:
    gripper_index = _gripper_index(names)
    return [index for index in range(len(names)) if index != gripper_index]


def _format_named_values(names: list[str], values: np.ndarray, indices: list[int]) -> str:
    if not indices:
        return "(无位置维度)"
    return ", ".join(
        f"{names[index]}={float(values[index]):+.4f}"
        for index in indices
    )


def _format_gripper_value(names: list[str], values: np.ndarray) -> str:
    gripper_index = _gripper_index(names)
    if gripper_index is None:
        return "(无 gripper 维度)"
    value = float(values[gripper_index])
    status = "打开" if value > 0.5 else "闭合"
    return f"{value:.4f} ({status})"


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs - rhs)))


def _episode_vector_pair_from_raw(raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(raw_path, allow_pickle=True) as data:
        tcp = np.asarray(data["tcp_poses"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32).reshape(-1, 1)
    state = np.concatenate([tcp[:-1], gripper[:-1]], axis=1).astype(np.float32)
    action = np.concatenate([tcp[1:], gripper[1:]], axis=1).astype(np.float32)
    return state, action


def _parquet_path_for_episode(root: Path, episode_index: int) -> Path:
    chunk_index = episode_index // 1000
    return root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"


def _load_task_map(root: Path) -> dict[int, str]:
    task_map: dict[int, str] = {}
    tasks_path = root / "meta" / "tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            task_map[int(payload["task_index"])] = str(payload["task"])
    return task_map


def _dataset_episode_vectors(root: Path, ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
    parquet_path = _parquet_path_for_episode(root, ep_idx)
    table = pq.read_table(parquet_path, columns=["observation.state", "action"])
    state_rows = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action_rows = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    return state_rows, action_rows


def _dataset_first_transition(root: Path, ep_idx: int, task_map: dict[int, str]) -> tuple[np.ndarray, np.ndarray, str]:
    parquet_path = _parquet_path_for_episode(root, ep_idx)
    table = pq.read_table(parquet_path, columns=["observation.state", "action", "task_index"])
    row = table.slice(0, 1).to_pydict()
    state = np.asarray(row["observation.state"][0], dtype=np.float32)
    action = np.asarray(row["action"][0], dtype=np.float32)
    task_index = int(row["task_index"][0])
    return state, action, task_map[task_index]


def _print_episode_comparison_summary(
    *,
    file_name: str,
    episode_index: int,
    state_diff: np.ndarray,
    action_diff: np.ndarray,
) -> None:
    state_max = float(state_diff.max()) if state_diff.size else 0.0
    action_max = float(action_diff.max()) if action_diff.size else 0.0
    print(
        f"[结果] {file_name} -> episode {episode_index}: "
        f"observation.state max_abs_diff={state_max:.6g}, "
        f"action max_abs_diff={action_max:.6g}"
    )
    print(
        "  observation.state 各维最大绝对误差: "
        + ", ".join(f"{value:.6g}" for value in state_diff.tolist())
    )
    print(
        "  action 各维最大绝对误差: "
        + ", ".join(f"{value:.6g}" for value in action_diff.tolist())
    )


def _print_transition_comparison(
    *,
    file_name: str,
    episode_index: int,
    state_names: list[str],
    action_names: list[str],
    state_expected: np.ndarray,
    state_actual: np.ndarray,
    action_expected: np.ndarray,
    action_actual: np.ndarray,
) -> None:
    state_position_indices = _position_indices(state_names)
    action_position_indices = _position_indices(action_names)

    print(f"[对比] {file_name} -> episode {episode_index} 的首条 transition")
    print(
        "  observation.state TCP(raw):     "
        f"{_format_named_values(state_names, state_expected, state_position_indices)}"
    )
    print(
        "  observation.state TCP(convert): "
        f"{_format_named_values(state_names, state_actual, state_position_indices)}"
    )
    print(
        "  observation.state 夹爪(raw):     "
        f"{_format_gripper_value(state_names, state_expected)}"
    )
    print(
        "  observation.state 夹爪(convert): "
        f"{_format_gripper_value(state_names, state_actual)}"
    )
    print(
        "  action TCP(raw):     "
        f"{_format_named_values(action_names, action_expected, action_position_indices)}"
    )
    print(
        "  action TCP(convert): "
        f"{_format_named_values(action_names, action_actual, action_position_indices)}"
    )
    print(
        "  action 夹爪(raw):     "
        f"{_format_gripper_value(action_names, action_expected)}"
    )
    print(
        "  action 夹爪(convert): "
        f"{_format_gripper_value(action_names, action_actual)}"
    )
    print(
        "  max_abs_diff: "
        f"observation.state={_max_abs_diff(state_expected, state_actual):.6g}, "
        f"action={_max_abs_diff(action_expected, action_actual):.6g}"
    )


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
    if raw_dir is not None and not raw_dir.exists():
        raise SystemExit(
            f"[失败] 原始 raw_demos 目录不存在: {raw_dir}\n"
            "请传入正确的 --raw-dir，以便比对转换前后的 TCP 和 gripper。"
        )

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
    state_names = _schema_names(
        manifest,
        dataset,
        feature_key="observation.state",
        manifest_key="state_names",
    )
    action_names = _schema_names(
        manifest,
        dataset,
        feature_key="action",
        manifest_key="action_names",
    )
    print(f"[信息] observation.state names={state_names}")
    print(f"[信息] action names={action_names}")

    if manifest is not None:
        specs = _camera_specs_from_manifest(manifest, dataset.fps)
        validate_existing_dataset(
            dataset,
            fps=dataset.fps,
            camera_specs=specs,
            state_names=state_names,
            action_names=action_names,
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
        task_map = _load_task_map(root)
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
            state_expected, action_expected, task_expected = _expected_transition_from_raw(raw_dir / file_name)
            state_actual, action_actual, task_actual = _dataset_first_transition(root, ep_idx, task_map)
            if state_actual.shape != state_expected.shape:
                raise SystemExit(
                    f"[失败] {file_name} 的 observation.state shape 不一致: "
                    f"raw={state_expected.shape}, converted={state_actual.shape}"
                )
            if action_actual.shape != action_expected.shape:
                raise SystemExit(
                    f"[失败] {file_name} 的 action shape 不一致: "
                    f"raw={action_expected.shape}, converted={action_actual.shape}"
                )
            if not np.allclose(state_actual, state_expected, atol=1e-5):
                raise SystemExit(f"[失败] {file_name} 的首条 observation.state 与 raw_demos 不一致")
            if not np.allclose(action_actual, action_expected, atol=1e-5):
                raise SystemExit(f"[失败] {file_name} 的首条 action 与 raw_demos 不一致")
            if task_actual != task_expected:
                raise SystemExit(f"[失败] {file_name} 的 task 与 raw_demos 不一致")
            _print_transition_comparison(
                file_name=file_name,
                episode_index=ep_idx,
                state_names=state_names,
                action_names=action_names,
                state_expected=state_expected,
                state_actual=state_actual,
                action_expected=action_expected,
                action_actual=action_actual,
            )

            expected_state_rows, expected_action_rows = _episode_vector_pair_from_raw(raw_dir / file_name)
            actual_state_rows, actual_action_rows = _dataset_episode_vectors(root, ep_idx)
            if actual_state_rows.shape != expected_state_rows.shape:
                raise SystemExit(
                    f"[失败] {file_name} 的整条 observation.state shape 不一致: "
                    f"raw={expected_state_rows.shape}, converted={actual_state_rows.shape}"
                )
            if actual_action_rows.shape != expected_action_rows.shape:
                raise SystemExit(
                    f"[失败] {file_name} 的整条 action shape 不一致: "
                    f"raw={expected_action_rows.shape}, converted={actual_action_rows.shape}"
                )
            if not np.allclose(actual_state_rows, expected_state_rows, atol=1e-5):
                raise SystemExit(f"[失败] {file_name} 的整条 observation.state 与 raw_demos 不一致")
            if not np.allclose(actual_action_rows, expected_action_rows, atol=1e-5):
                raise SystemExit(f"[失败] {file_name} 的整条 action 与 raw_demos 不一致")

            state_diff = np.max(np.abs(actual_state_rows - expected_state_rows), axis=0)
            action_diff = np.max(np.abs(actual_action_rows - expected_action_rows), axis=0)
            _print_episode_comparison_summary(
                file_name=file_name,
                episode_index=ep_idx,
                state_diff=state_diff,
                action_diff=action_diff,
            )

        print(f"[通过] 抽样检查了前 {len(checks)} 条 episode 的 state/action/task 数值")

    print("[信息] 已通过 meta/info.json、parquet 和 mp4 文件数量完成图像结构校验；未执行视频解码抽样。")

    print("[完成] 转换后的 LeRobotDataset v2.1 通过检查")


if __name__ == "__main__":
    main()
