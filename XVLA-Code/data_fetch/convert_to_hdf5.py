"""
将采集的 .npz 轨迹转换为 X-VLA 训练用的 HDF5 格式（robomind-ur 格式）

输入: ./raw_demos/episode_XXXX.npz
  - images:          (T, 256, 256, 3) uint8  主视角
  - images_wrist:    (T, 256, 256, 3) uint8  腕部视角（可选）
  - tcp_poses:       (T, 6) [x, y, z, rx, ry, rz] axis-angle (RTDE 返回)
  - gripper:         (T, 1) float
  - instruction:     str

输出: ./training_data/demo_XX.hdf5
  - observations/images/cam_high:   (T, 256, 256, 3) uint8
  - observations/images/cam_wrist:  (T, 256, 256, 3) uint8（如果有的话）
  - puppet/end_effector:            (T, 6) float32 [x,y,z, euler_x,euler_y,euler_z]
  - puppet/joint_position:          (T, 7) float32 [6关节 + gripper]
  - language_instruction:           str
"""

import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.transform import Rotation

INPUT_DIR = Path("./raw_demos")
OUTPUT_DIR = Path("./training_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def axisangle_to_euler_xyz(rvec):
    """
    轴角 (RTDE 返回) → euler XYZ (robomind-ur 格式)

    RTDE getActualTCPPose() 返回的旋转是轴角表示 [rx, ry, rz]
    robomind-ur 的 end_effector 使用 euler 'xyz' 约定
    """
    R = Rotation.from_rotvec(rvec)
    return R.as_euler("xyz")  # 返回 3 维 euler 角


def main():
    npz_files = sorted(INPUT_DIR.glob("*.npz"))
    print(f"找到 {len(npz_files)} 条轨迹\n")

    total_frames = 0
    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        images = data["images"]         # (T, 256, 256, 3) uint8
        tcp_poses = data["tcp_poses"]   # (T, 6) [x, y, z, rx, ry, rz] axis-angle
        gripper = data["gripper"]       # (T,) float
        instruction = str(data["instruction"])

        # 腕部视角（可选，双摄像头采集时有）
        images_wrist = data.get("images_wrist", None)

        T = len(images)
        total_frames += T

        # ---- 转换旋转表示：轴角 → euler xyz ----
        eulers = np.array([axisangle_to_euler_xyz(tcp[3:6]) for tcp in tcp_poses])  # (T, 3)

        # 组装 end_effector: [x, y, z, euler_x, euler_y, euler_z]
        end_effector = np.concatenate([tcp_poses[:, :3], eulers], axis=1).astype(np.float32)  # (T, 6)

        # 组装 joint_position: 6 个关节 + gripper
        # 由于我们没有记录关节角度（只记录了末端位姿），这里填入零 + gripper
        # X-VLA 的 robomind-ur handler 只用 end_effector 和 gripper，关节角度不参与训练
        joint_position = np.zeros((T, 7), dtype=np.float32)
        joint_position[:, -1] = gripper.flatten()  # 最后一列 = gripper

        # ---- 保存为 HDF5 ----
        out_path = OUTPUT_DIR / f"demo_{i:04d}.hdf5"
        with h5py.File(out_path, "w") as f:
            # 图像
            obs_grp = f.create_group("observations")
            img_grp = obs_grp.create_group("images")
            img_grp.create_dataset("cam_high", data=images, dtype=np.uint8, compression="gzip")
            if images_wrist is not None:
                img_grp.create_dataset("cam_wrist", data=images_wrist, dtype=np.uint8, compression="gzip")

            # 位姿
            puppet_grp = f.create_group("puppet")
            puppet_grp.create_dataset("end_effector", data=end_effector)
            puppet_grp.create_dataset("joint_position", data=joint_position)

            # 语言指令
            f.attrs["language_instruction"] = instruction

        print(f"  [{i+1}/{len(npz_files)}] {out_path.name} — {T} 帧 "
              f"xyz=[{tcp_poses[0,0]:.3f},{tcp_poses[0,1]:.3f},{tcp_poses[0,2]:.3f}]")

    print(f"\n转换完成: {len(npz_files)} 条轨迹, 共 {total_frames} 帧")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()