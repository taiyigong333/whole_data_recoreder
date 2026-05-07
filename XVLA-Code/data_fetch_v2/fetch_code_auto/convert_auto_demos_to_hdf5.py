from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from config_utils import DEFAULT_CONFIG_PATH, load_config, resolve_project_path


def axisangle_to_euler_xyz(rotvec: np.ndarray) -> np.ndarray:
    """把 RTDE 的轴角旋转表示转换为 euler xyz。"""
    return Rotation.from_rotvec(rotvec).as_euler("xyz")


def build_joint_state(data: np.lib.npyio.NpzFile, frame_count: int) -> np.ndarray:
    """组装训练所需的 7 维 joint_state。

    前 6 维放机械臂关节角，第 7 维放夹爪开合值。
    """
    joint_state = np.zeros((frame_count, 7), dtype=np.float32)

    if "joint_positions" in data.files:
        joints = np.asarray(data["joint_positions"], dtype=np.float32)
        if joints.ndim == 2 and joints.shape[1] >= 6:
            joint_state[:, :6] = joints[:, :6]

    gripper = np.asarray(data["gripper"], dtype=np.float32).reshape(-1)
    joint_state[:, 6] = gripper[:frame_count]
    return joint_state


def convert_one(npz_path: Path, output_dir: Path, demo_index: int) -> Path:
    """把单个 npz 轨迹转换为训练所需的 HDF5 文件。"""
    data = np.load(npz_path, allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)
    images_wrist = np.asarray(data["images_wrist"], dtype=np.uint8)
    tcp_poses = np.asarray(data["tcp_poses"], dtype=np.float64)
    instruction = str(data["instruction"])

    frame_count = images.shape[0]
    eulers = np.asarray(
        [axisangle_to_euler_xyz(pose[3:6]) for pose in tcp_poses],
        dtype=np.float32,
    )
    end_effector = np.concatenate(
        [tcp_poses[:, :3].astype(np.float32), eulers],
        axis=1,
    )
    joint_position = build_joint_state(data, frame_count)

    output_path = output_dir / f"demo_{demo_index:04d}.hdf5"
    with h5py.File(output_path, "w") as handle:
        obs_group = handle.create_group("observations")
        img_group = obs_group.create_group("images")
        img_group.create_dataset("cam_high", data=images, dtype=np.uint8, compression="gzip")
        img_group.create_dataset("cam_wrist", data=images_wrist, dtype=np.uint8, compression="gzip")

        puppet_group = handle.create_group("puppet")
        puppet_group.create_dataset("end_effector", data=end_effector)
        puppet_group.create_dataset("joint_position", data=joint_position)

        handle.attrs["language_instruction"] = instruction
        if "task_name" in data.files:
            handle.attrs["task_name"] = str(data["task_name"])
        if "metadata_json" in data.files:
            handle.attrs["metadata_json"] = str(data["metadata_json"])

    return output_path


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="配置文件路径，默认使用 data_fetch_v2/configs/auto_collect_config.json",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="待转换的 npz 目录；为空时使用配置文件中的 save_dir",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出 HDF5 目录；为空时使用配置文件中的 training_data_dir",
    )
    return parser


def main() -> None:
    """程序入口。"""
    args = build_parser().parse_args()
    runtime_config, _ = load_config(args.config)

    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
    else:
        input_dir = resolve_project_path(runtime_config["paths"]["save_dir"])

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = resolve_project_path(runtime_config["paths"]["training_data_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("episode_*.npz"))
    print(f"在 {input_dir} 中找到 {len(npz_files)} 个轨迹文件")

    total_frames = 0
    for demo_index, npz_path in enumerate(npz_files):
        output_path = convert_one(npz_path, output_dir, demo_index)
        frame_count = int(np.load(npz_path, allow_pickle=True)["images"].shape[0])
        total_frames += frame_count
        print(f"[{demo_index + 1}/{len(npz_files)}] {output_path.name} <- {npz_path.name} ({frame_count} 帧)")

    print(f"转换完成，共处理 {len(npz_files)} 个文件，总帧数 {total_frames}")


if __name__ == "__main__":
    main()
