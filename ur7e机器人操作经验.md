# UR7e 数据采集初步尝试

> 本文档面向首次接触 UR7e 机器人数据采集。
> 内容覆盖从硬件接线到采集出第一条轨迹的完整流程。
>
> **前置知识**：基本 Python 操作、conda 环境管理。

---

## 目录

1. [硬件接线](#1-硬件接线)
2. [网络配置](#2-网络配置)
3. [示教器设置](#3-示教器设置)
4. [安装 Python 依赖](#4-安装-python-依赖)
5. [测试连接 — 读取位姿](#5-测试连接--读取位姿)
6. [测试连接 — 控制运动（可选，需远程控制）](#6-测试连接--控制运动可选需远程控制)
7. [诊断夹爪](#7-诊断夹爪)
8. [采集原理](#8-采集原理)
9. [运行采集脚本](#9-运行采集脚本)
10. [采集操作流程](#10-采集操作流程)
11. [采集数据检查](#11-采集数据检查)
12. [常见问题](#常见问题)

---

## 1. 硬件接线

需要 2 根线：

```
                          ┌──────────────────┐
                          │    UR7e 控制柜    │
                          │                  │
                          │  以太网口         │
                          └───────┬──────────┘
                                  │ 网线
                                  ↓
┌──────────┐              ┌──────────────────┐
│ USB 相机  │──USB 线──→  │     PC / 笔记本    │
└──────────┘              │                  │
                          │  网口      USB 口  │
                          └──────────────────┘
```

- **网线**：PC 网口 ↔ UR7e 控制柜以太网口
- **USB 线**：USB 相机 ↔ PC
- 相机用三脚架固定在桌面旁，朝向机器人工作区域

## 2. 网络配置

UR7e 控制柜默认 IP 为 `192.168.1.88`（示教器上可查看）。

给 PC 配同网段静态 IP：

**Windows 11：**
1. 设置 → 网络和 Internet → 以太网 → IP 设置 → 编辑
2. 选择"手动"，打开 IPv4
3. 填写：
   - IP 地址：`192.168.1.88`
   - 子网掩码：`255.255.255.0`
   - 默认网关、DNS：**留空**
4. 保存

验证：
```bash
ping 192.168.1.88
# 应收到回复，延迟 <1ms
```

## 3. 示教器设置

在 UR7e 示教器（触摸屏）上：

1. **开机**：打开控制柜电源开关，等待示教器启动
2. **初始化**：按提示完成开机初始化（"开启机器人" → 确认安全）
3. **切换模式**：
   - **采集阶段**（步骤 9-10）：设置为 **"本地手动模式"**，脚本只读取数据不发送指令
   - **测试运动**（步骤 6，可选）：需要临时切换到 **"远程控制 (Remote Control)"** 模式
4. **检查 IP**：设置 → 网络 → 确认 IP 为 `192.168.1.88`

## 4. 安装 Python 依赖

```bash
conda create -n ur7e_collect python=3.10 -y
conda activate ur7e_collect

pip install ur-rtde          # UR 机器人 RTDE 通信
pip install opencv-python    # 相机采集
pip install numpy scipy      # 数值计算
```

## 5. 测试连接 — 读取位姿

新建 `test_connection.py`：

```python
# test_connection.py
from rtde_receive import RTDEReceiveInterface

rtde_r = RTDEReceiveInterface("192.168.1.88")

tcp = rtde_r.getActualTCPPose()
q = rtde_r.getActualQ()

print(f"末端位姿 [x, y, z, rx, ry, rz]: {tcp}")
print(f"关节角度: {q}")
print("连接成功！")
```

运行：
```bash
python test_connection.py
```

期望输出：
```
末端位姿 [x, y, z, rx, ry, rz]: [-0.2, -0.3, 0.5, 0.0, 3.14, 0.0]
关节角度: [-1.57, -1.57, 1.57, -1.57, -1.57, 0.0]
连接成功！
```

> 如果报错 `Connection refused`，检查：网线是否插好、IP 是否配对、示教器是否开机完成。

## 6. 测试连接 — 控制运动（可选，需远程控制）

> **注意**：此测试需要示教器切换到"远程控制"模式。如果只想验证读取是否正常，跳过此步即可。采集数据时不需要远程控制。

新建 `test_move.py`：

```python
# test_move.py
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time

rtde_c = RTDEControlInterface("192.168.1.88")
rtde_r = RTDEReceiveInterface("192.168.1.88")

# 读取当前位姿
current = rtde_r.getActualTCPPose()
print(f"当前位置: x={current[0]:.3f}, y={current[1]:.3f}, z={current[2]:.3f}")

# 沿 Z 轴上移 2cm
target = current.copy()
target[2] += 0.02  # z + 2cm
print(f"目标位置: x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f}")

input("确认安全后按 Enter 执行移动（速度很慢）...")
rtde_c.moveL(target, speed=0.02, acceleration=0.02)  # 极慢：2cm/s

time.sleep(1)
new_pose = rtde_r.getActualTCPPose()
print(f"移动后: x={new_pose[0]:.3f}, y={new_pose[1]:.3f}, z={new_pose[2]:.3f}")

rtde_c.stopScript()
```

## 7. 诊断夹爪

采集前必须确认夹爪状态通过哪个 RTDE 寄存器读取。新建 `diagnose_gripper.py`：

```python
# diagnose_gripper.py
"""
诊断夹爪状态寄存器
运行一次 → 手动开合夹爪 → 再运行一次 → 对比哪个值变了
"""
from rtde_receive import RTDEReceiveInterface

rtde_r = RTDEReceiveInterface("192.168.1.88")

print("=== 数字输出 (Digital Output) ===")
dout = rtde_r.getActualDigitalOutBits()
print(f"  DO bits: {dout}  (二进制: {bin(dout)})")
for i in range(8):
    print(f"  DO[{i}] = {(dout >> i) & 1}")

print("\n=== 数字输入 (Digital Input) ===")
din = rtde_r.getActualDigitalInputBits()
print(f"  DI bits: {din}  (二进制: {bin(din)})")
for i in range(8):
    print(f"  DI[{i}] = {(din >> i) & 1}")

print("\n=== 工具数字输出 (Tool Digital Output) ===")
try:
    tdout = rtde_r.getToolDigitalOutBits()
    print(f"  Tool DO bits: {tdout}")
    for i in range(2):
        print(f"  Tool DO[{i}] = {(tdout >> i) & 1}")
except Exception as e:
    print(f"  不可用: {e}")

print("\n=== 工具数字输入 (Tool Digital Input) ===")
try:
    tdin = rtde_r.getToolDigitalInputBits()
    print(f"  Tool DI bits: {tdin}")
    for i in range(2):
        print(f"  Tool DI[{i}] = {(tdin >> i) & 1}")
except Exception as e:
    print(f"  不可用: {e}")

print("\n" + "=" * 50)
print("请现在在示教器上手动开合夹爪，然后再次运行此脚本")
print("对比两次输出，找到变化的那个寄存器")
print("=" * 50)
```

**操作步骤：**
1. 运行 `python diagnose_gripper.py`，记录所有值
2. 在示教器上让夹爪**闭合**
3. 再次运行 `python diagnose_gripper.py`
4. 对比哪个值变了 → 记下来，采集脚本要用

**常见结果：**

| 夹爪类型 | 变化的寄存器 | 闭合时的值 |
|---------|------------|-----------|
| Robotiq 2F-85 | `DO[0]` 或 `Tool DO[0]` | 1 |
| OnRobot | `DO[0]` | 1 |
| 气动夹爪 | `DO[0]` | 1 |

> 到这一步，如果位姿能读、夹爪状态能读，硬件连接就没问题了，可以开始采集。

---

## 8. 采集原理

不管哪种采集方案，核心逻辑相同：

```
每帧循环（30 FPS）：
  ① 并行：RTDE 读位姿 + 相机拍照（线程并行，缩小时间差）
  ② RTDE 读取夹爪状态（自动，无需手动标记）
  ③ 保存 (image, tcp_pose, gripper)
  ④ 下一帧...

一条轨迹结束后保存为一个 .npz 文件。
```

数据内容包括：
- **图像**：256×256 RGB（相机原始帧缩放而来）
- **TCP 位姿**：6 维 `[x, y, z, rx, ry, rz]`（RTDE 读取，axis-angle 格式）
- **夹爪状态**：0.0 = 打开，1.0 = 闭合

## 9. 运行采集脚本

### 9.1 配置

根据第 7 步的诊断结果，修改脚本顶部的 `GRIPPER_SOURCE`：

```python
# 诊断发现 DO[0] 变化（最常见）：
GRIPPER_SOURCE = "do"
GRIPPER_INDEX = 0

# 诊断发现 Tool DO[0] 变化：
# GRIPPER_SOURCE = "tool_do"
# GRIPPER_INDEX = 0

# 诊断发现 DI[0] 变化：
# GRIPPER_SOURCE = "di"
# GRIPPER_INDEX = 0
```

如果不确定夹爪类型或诊断失败，设为 `"manual"`，采集时用键盘标记。

### 9.2 完整采集脚本

```python
# collect_freedrive.py
"""
UR7e 自由拖动示教数据采集
仅需 RTDEReceiveInterface（只读），不需要远程控制权限
使用前：示教器上切换到本地手动模式，开启自由驱动（Freedrive）
"""

import cv2
import time
import threading
import numpy as np
from pathlib import Path
from rtde_receive import RTDEReceiveInterface

# ============ 配置（根据你的环境修改） ============
ROBOT_IP = "192.168.1.88"       # UR7e 控制柜 IP
CAMERA_INDEX = 1                # USB 相机设备号
CAMERA_WIDTH = 640              # 采集分辨率
CAMERA_HEIGHT = 480
SAVE_DIR = Path("./raw_demos") # 原始数据保存目录
FPS = 30                        # 采集帧率（与 robomind-ur 训练数据一致）
TASK_NAME = "pick up the red block"  # 语言指令
MAX_FRAMES = 900                # 单条轨迹最大帧数（30fps × 30秒）

# 夹爪配置 —— 运行 diagnose_gripper.py 后修改此项
# 可选值: "do" (数字输出), "di" (数字输入), "tool_do" (工具数字输出), "tool_di" (工具数字输入), "manual" (手动键盘标记)
GRIPPER_SOURCE = "do"
GRIPPER_INDEX = 0               # 寄存器编号（通常是 0）
# ==============================

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def find_camera(camera_index=CAMERA_INDEX):
    """优先使用指定设备号；失败后再自动搜索"""
    # 先尝试指定设备
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"使用指定相机: 设备号 {camera_index}")
            return cap, camera_index
        cap.release()

    # 指定失败后再搜索
    for idx in range(0, 6):
        if idx == camera_index:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"找到相机: 设备号 {idx}")
                return cap, idx
            cap.release()

    raise RuntimeError("未找到可用相机，请检查 USB 连接")


class GripperReader:
    """自动读取夹爪状态（根据 diagnose_gripper.py 诊断结果配置）"""

    def __init__(self, rtde_r, source=GRIPPER_SOURCE, index=GRIPPER_INDEX):
        self.rtde_r = rtde_r
        self.source = source
        self.index = index
        print(f"夹爪读取方式: {source}[{index}]")

    def read(self):
        """返回夹爪状态: 0.0=打开, 1.0=闭合"""
        try:
            if self.source == "do":
                bits = self.rtde_r.getActualDigitalOutBits()
                return float((bits >> self.index) & 1)
            elif self.source == "di":
                bits = self.rtde_r.getActualDigitalInputBits()
                return float((bits >> self.index) & 1)
            elif self.source == "tool_do":
                bits = self.rtde_r.getToolDigitalOutBits()
                return float((bits >> self.index) & 1)
            elif self.source == "tool_di":
                bits = self.rtde_r.getToolDigitalInputBits()
                return float((bits >> self.index) & 1)
            else:
                return -1.0
        except Exception:
            return -1.0


def capture_aligned(rtde_r, cap):
    """并行采集图像和位姿，尽量缩小时间差"""
    pose_holder = [None]

    def read_pose():
        pose_holder[0] = rtde_r.getActualTCPPose()

    # 先启动位姿读取线程（RTDE 延迟 ~2ms，比相机快）
    t = threading.Thread(target=read_pose)
    t.start()

    # 同时拍照（OpenCV 延迟 ~10-30ms）
    ret, frame = cap.read()

    # 等位姿读完
    t.join()

    if not ret or pose_holder[0] is None:
        return None, None
    return frame, pose_holder[0]


def main():
    # ---- 初始化（只连接接收接口，不发送控制指令）----
    print("正在连接机器人（只读模式）...")
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    print(f"机器人已连接，当前位姿: {rtde_r.getActualTCPPose()}")

    cap, cam_idx = find_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    gripper_reader = GripperReader(rtde_r)

    episode = 0
    interval = 1.0 / FPS

    # 检测夹爪状态是否可用
    test_grip = gripper_reader.read()
    if test_grip < 0:
        print("\n⚠ 夹爪自动读取失败！回退到手动键盘标记模式")
        print("  录制时按 'c' = 闭合，按 'g' = 打开")
        gripper_reader.source = "manual"
    else:
        print(f"夹爪当前状态: {'闭合' if test_grip > 0.5 else '打开'} ✓")

    print("\n" + "=" * 50)
    print("UR7e 自由拖动采集（只读模式）")
    print("=" * 50)
    print("操作步骤：")
    print("  0. 示教器上进入 本地手动模式 → 开启自由驱动(Freedrive)")
    print("  1. 按住示教器背面按钮，手拖机器人")
    print("  2. 通过示教器控制夹爪开合（脚本自动读取状态）")
    if gripper_reader.source == "manual":
        print("  2b. 按 'c' 标记夹爪闭合，按 'g' 标记打开")
    print("  3. 按 'q' 或 ESC 结束当前轨迹")
    print("  4. 按 Enter 开始下一条")
    print("=" * 50)

    last_manual_grip = 1.0

    try:
        while True:
            print(f"\n--- 轨迹 #{episode} ---")
            print("将机器人移到起始位置，示教器开启自由驱动")
            input("准备好后按 Enter 开始录制...")

            images = []
            tcp_poses = []
            gripper_states = []
            frame_count = 0

            while frame_count < MAX_FRAMES:
                loop_start = time.time()

                # ① 并行采集：图像 + 位姿
                frame, tcp = capture_aligned(rtde_r, cap)
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (256, 256))
                images.append(frame_resized)
                tcp_poses.append(tcp)

                # ② 读取夹爪状态
                grip = gripper_reader.read()
                if grip < 0:
                    grip = last_manual_grip
                gripper_states.append(grip)

                frame_count += 1

                # ③ 显示预览
                display = frame.copy()
                cv2.putText(display, f"Ep#{episode} F:{frame_count} FPS:{FPS}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display,
                            f"TCP: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                grip_str = "CLOSED" if grip > 0.5 else "OPEN"
                grip_color = (0, 0, 255) if grip > 0.5 else (0, 255, 0)
                cv2.putText(display, f"Gripper: {grip_str}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 2)

                if gripper_reader.source == "manual":
                    cv2.putText(display, "Manual: C=close G=open | Q=stop",
                                (10, 480-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(display, "Auto gripper | Q=stop",
                                (10, 480-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.imshow("UR7e Collection", display)

                key = cv2.waitKey(1) & 0xFF

                if gripper_reader.source == "manual":
                    if key == ord('c'):
                        last_manual_grip = 1.0
                        gripper_states[-1] = 1.0
                        print("  夹爪 → 闭合")
                    elif key == ord('g'):
                        last_manual_grip = 0.0
                        gripper_states[-1] = 0.0
                        print("  夹爪 → 打开")

                if key == ord('q') or key == 27:
                    break

                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

            # ---- 保存轨迹 ----
            if len(images) < 30:
                print(f"轨迹太短（{len(images)} 帧），丢弃")
                continue

            images_arr = np.array(images, dtype=np.uint8)
            tcp_arr = np.array(tcp_poses, dtype=np.float64)
            grip_arr = np.array(gripper_states, dtype=np.float64)

            save_path = SAVE_DIR / f"episode_{episode:04d}.npz"
            np.savez_compressed(
                save_path,
                images=images_arr,
                tcp_poses=tcp_arr,
                gripper=grip_arr,
                instruction=TASK_NAME,
                fps=FPS,
            )
            print(f"已保存: {save_path} ({len(images)} 帧, "
                  f"gripper 闭合帧数: {int(grip_arr.sum())})")

            episode += 1

            cont = input("继续采集下一条？(y/n): ")
            if cont.lower() != 'y':
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"采集结束，共保存 {episode} 条轨迹到 {SAVE_DIR}")


if __name__ == "__main__":
    main()
```

## 10. 采集操作流程

以"抓起桌上的红色方块"为例：

```
第 1 条轨迹：
  1. 示教器上开启自由驱动（Freedrive）
  2. 运行 python collect_freedrive.py
  3. 将红色方块放在桌面上固定位置
  4. 按 Enter 开始录制
  5. 按住示教器背面按钮，手拖机器人：
     a. 移动到方块上方
     b. 下降到方块旁边
     c. 在示教器上控制夹爪闭合  ← 脚本自动读到夹爪闭合
     d. 提起方块
     e. 移动到目标位置
     f. 在示教器上控制夹爪打开  ← 脚本自动读到夹爪打开
     g. 放下方块
  6. 按 'q' 结束本条轨迹 → 自动保存到 ./raw_demos/
  7. 将方块放回原位
  8. 输入 'y' 开始下一条

重复以上步骤，采集 50-100 条。
```

> 如果夹爪自动读取失败（脚本会提示并回退到手动模式），按键盘 'c' 标记闭合、'g' 标记打开。

---

## 11. 采集数据检查

采集完成后可以快速检查数据质量：

```python
# check_data.py
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
```

---

## 脚本清单

| 文件 | 用途 | 何时运行 |
|------|------|---------|
| `test_connection.py` | 测试 RTDE 连接、读取位姿 | 首次连接时 |
| `test_move.py` | 测试 RTDE 控制运动（需远程控制模式） | 首次连接时（可选） |
| `diagnose_gripper.py` | 诊断夹爪寄存器 | 首次连接时（采集前必跑） |
| `collect_freedrive.py` | 采集示教轨迹 | 采集阶段（主脚本） |
| `check_data.py` | 检查采集数据质量 | 采集完成后 |

## 常见问题

**Q: `Connection refused` 或 `Connection timed out`**
- 检查网线是否插好
- 检查 PC 的 IP 是否配了 `192.168.1.10`，子网掩码 `255.255.255.0`
- 在 PC 上 `ping 192.168.1.88` 看是否通

**Q: "未找到可用相机"**
- 检查 USB 相机是否插好
- 尝试修改 `CAMERA_INDEX`（0 → 1 → 2）

**Q: 夹爪状态始终不变**
- 重新运行 `diagnose_gripper.py`，确认寄存器
- 如果自动读取确实不行，设置 `GRIPPER_SOURCE = "manual"` 用键盘标记

**Q: 采集时机器人不动**
- 正常现象。采集脚本只读取数据，不控制运动
- 机器人需要人在示教器上手动拖动