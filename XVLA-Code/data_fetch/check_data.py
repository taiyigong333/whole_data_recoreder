import numpy as np
from pathlib import Path

demo_dir = Path("./raw_demos")
files = sorted(demo_dir.glob("*.npz"))

print(f"共 {len(files)} 条轨迹\n")

lengths = []
for f in files:
    data = np.load(f)
    length = len(data["images"])
    lengths.append(length)
    tcp = data["tcp_poses"]
    grip = data["gripper"]
    closed_frames = int(grip.sum())
    print(f"  {f.name}: {length} 帧, "
          f"起始 {[f'{v:.3f}' for v in tcp[0, :3]]}, "
          f"终止 {[f'{v:.3f}' for v in tcp[-1, :3]]}, "
          f"夹爪闭合: {closed_frames}/{length} 帧")

print(f"\n平均长度: {np.mean(lengths):.0f} 帧")
print(f"最短: {min(lengths)}, 最长: {max(lengths)}")

# 检查夹爪状态是否有变化（如果全是同一个值说明读取有问题）
all_same = True
for f in files[:5]:
    data = np.load(f)
    grip = data["gripper"]
    if grip.min() != grip.max():
        all_same = False
        break
if all_same:
    print("\n⚠ 警告：所有轨迹的夹爪状态始终不变，可能读取方式有误")
    print("  请重新运行 diagnose_gripper.py 检查夹爪寄存器")

# 运动范围
data = np.load(files[0])
tcp = data["tcp_poses"]
print(f"\n第 1 条轨迹 XYZ 范围:")
print(f"  X: [{tcp[:,0].min():.3f}, {tcp[:,0].max():.3f}]")
print(f"  Y: [{tcp[:,1].min():.3f}, {tcp[:,1].max():.3f}]")
print(f"  Z: [{tcp[:,2].min():.3f}, {tcp[:,2].max():.3f}]")