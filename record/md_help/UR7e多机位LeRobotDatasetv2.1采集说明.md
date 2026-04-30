# UR7e 多机位采集到 LeRobotDataset v2.1

本文档对应当前仓库里的采集入口：

- `record/collect_ur7e_multicam_lerobot_v21.py`

配套实现已经拆到：

- `record/ur7e_lerobot_recorder/`

现在这套方案有三个核心变化：

1. 相机层不再手写 `OpenCV VideoCapture`，而是统一复用 `lerobot` 官方相机抽象
   - `opencv`
   - `intelrealsense`
2. 一次录制到底用几路相机，不再写死在脚本里，而是由 `--camera-config` 决定
   - 1 路、2 路、3 路都可以
   - 但同一个数据集目录内，相机数量和相机键必须固定
3. 入口脚本已经瘦身，配置、相机、录制、数据集写入都拆成独立模块，后续更容易改

除此之外，入口脚本现在同时支持两类工作流：

- `record`
  - 直接采集并写成 `LeRobotDataset v2.1`
- `convert`
  - 将 `XVLA-Code/data_fetch` 采到的 `raw_demos/*.npz` 离线转换成 `LeRobotDataset v2.1`

---

## 1. 方案说明

### 1.1 机器人侧

UR7e 保持在：

- 本地手动模式
- Freedrive / 自由驱动

脚本只读取：

- `getActualQ()`
- `getActualTCPPose()`
- 数字 IO / 工具 IO 中的夹爪状态

不会向机器人发送运动命令。

### 1.2 相机侧

相机统一走 `lerobot v0.3.3` 自带的相机抽象：

- `opencv`
- `intelrealsense`

也就是说：

- 如果你接的是普通 USB 相机，配置里用 `opencv`
- 如果你接的是 Intel RealSense，配置里用 `intelrealsense`

### 1.3 数据集里 `action` 的定义

因为自由拖动示教没有“控制命令流”，但训练又必须有 `action`，所以脚本仍然采用 transition 形式：

- `observation.state`
  - 当前时刻关节角 `q1..q6`
  - 当前时刻 TCP 位姿 `tcp_x..tcp_rz`
  - 当前时刻夹爪状态 `gripper`
- `action`
  - 下一时刻绝对 TCP 位姿 `tcp_x..tcp_rz`
  - 下一时刻夹爪状态 `gripper`

也就是：

- 第 0 帧：`observation=s0`，`action=s1`
- 第 1 帧：`observation=s1`，`action=s2`
- ...

这和 `LeRobotDataset` 的 transition 组织方式是一致的。

---

## 2. 依赖安装

### 2.1 当前推荐环境

建议直接使用你已经有的 `lerobot` conda 环境：

```bash
conda activate lerobot
python -V
```

### 2.2 先装 `record` 的 requirements

仓库里现在单独提供了采集脚本的依赖清单：

- `record/requirements.txt`
- `record/requirements-realsense.txt`

推荐安装顺序：

```bash
cd /Users/gongtaiyi/code/vla_start/lerobot
conda activate lerobot

python -m pip install -e ./lerobot-0.3.3
python -m pip install -r record/requirements.txt
```

如果你需要 RealSense，再额外执行：

```bash
python -m pip install -r record/requirements-realsense.txt
```

这两个文件的职责是：

- `record/requirements.txt`
  - 补 `ur-rtde`
  - 把 `opencv-python-headless` 覆盖为支持预览窗口的 `opencv-python`
- `record/requirements-realsense.txt`
  - 在基础依赖之上补 RealSense 的 Python 依赖

### 2.3 `pyrealsense2` 的安装说明

截至 **2026-04-29**，`pyrealsense2` 在 PyPI 的 files 页面没有提供 macOS arm64 wheel。

当前这台机器是：

- macOS arm64
- Python 3.10

所以更稳的安装方式是：

```bash
conda install -n lerobot -c conda-forge pyrealsense2
```

也就是说：

- Linux 上可以优先试 `record/requirements-realsense.txt`
- macOS arm64 上优先走 conda，不建议只依赖 pip

### 2.4 `ur-rtde` 的安装说明

如果你没有用上面的 requirements 文件，也可以单独安装：

```bash
conda activate lerobot
python -m pip install ur-rtde
```

如果是在 macOS 上从源码编译，可能还需要：

- `cmake`
- `boost-cpp`

也就是说，如果你遇到 CMake / Boost 报错，优先补：

```bash
conda install -n lerobot -c conda-forge cmake boost-cpp
```

### 2.5 `ffmpeg`

检查：

```bash
conda activate lerobot
ffmpeg -version
```

`LeRobotDataset v2.1` 在 `use_videos=True` 时会把每个 episode 的临时帧编码成 mp4，所以 `ffmpeg` 必须可用。

---

## 3. 相机配置方式

### 3.1 为什么一定要单独配相机

一批 `LeRobotDataset` 的字段布局是固定的。

例如第一次建数据集时，如果你用了：

- `observation.images.front`
- `observation.images.left_arm`

那么后续 `--resume` 时也必须还是这两路，不能突然改成三路或一路。

所以现在脚本把“本次录制到底用哪些相机”收进了 JSON 配置文件里。

### 3.2 仓库里提供的示例配置

可以直接参考：

- `record/camera_configs/opencv_single.json`
- `record/camera_configs/opencv_dual.json`
- `record/camera_configs/opencv_triple.json`
- `record/camera_configs/realsense_dual.template.json`

### 3.3 配置文件格式

最小示例：

```json
{
  "cameras": [
    {
      "name": "front",
      "type": "opencv",
      "device": 0
    },
    {
      "name": "left_arm",
      "type": "opencv",
      "device": 1
    }
  ]
}
```

字段说明：

- `name`
  - 数据集里的相机键名
  - 最终会变成 `observation.images.<name>`
- `type`
  - `opencv`
  - `intelrealsense`
- `device`
  - 对 `opencv`：可以是整数设备号，也可以是视频路径
  - 对 `intelrealsense`：一般写序列号

可选字段：

- `width`
- `height`
- `fps`
- `rotation`
- `warmup_s`
- `use_depth`

如果配置里不单独写 `width/height`，就继承命令行里的：

- `--image-width`
- `--image-height`

---

## 4. 采集前准备

### 4.1 硬件连接

参考 `ur7e机器人操作经验.md`：

- PC 网口连接 UR7e 控制柜
- USB 相机或 RealSense 相机连接到 PC
- 相机固定到稳定视角

### 4.2 网络

确保 PC 与机器人在同一网段，例如：

- 机器人：`192.168.1.88`
- PC：`192.168.1.x`

先确认：

```bash
ping 192.168.1.88
```

### 4.3 机器人模式

采集时保持：

- 示教器进入“本地手动模式”
- 开启 Freedrive

---

## 5. 夹爪状态准备

仍然建议先按 `ur7e机器人操作经验.md` 里的方式诊断夹爪寄存器。

常见用法：

- `--gripper-source do --gripper-index 0`
- `--gripper-source di --gripper-index 0`
- `--gripper-source tool_do --gripper-index 0`
- `--gripper-source tool_di --gripper-index 0`

如果自动读取不稳定，可以切到人工模式：

```bash
--gripper-source manual
```

人工模式下：

- `c`：夹爪闭合
- `o` 或 `g`：夹爪打开

---

## 6. 先扫描相机

### 6.1 扫描 OpenCV 相机

```bash
cd /Users/gongtaiyi/code/vla_start/lerobot
conda activate lerobot

python record/collect_ur7e_multicam_lerobot_v21.py \
  --scan-cameras \
  --scan-camera-type opencv \
  --max-camera-index 10
```

你会看到类似输出：

```text
正在扫描 OpenCV 相机...
  id=0 backend=AVFOUNDATION default=640x480@30.0
  id=1 backend=AVFOUNDATION default=640x480@30.0
```

### 6.2 扫描 RealSense 相机

```bash
python record/collect_ur7e_multicam_lerobot_v21.py \
  --scan-cameras \
  --scan-camera-type intelrealsense
```

正常时会打印：

- 相机名字
- 序列号
- 默认分辨率和默认 fps

如果当前环境没有 `pyrealsense2`，脚本会直接提示并跳过 RealSense 扫描。

---

## 7. 开始采集

### 7.1 单相机采集

```bash
cd /Users/gongtaiyi/code/vla_start/lerobot
conda activate lerobot

python record/collect_ur7e_multicam_lerobot_v21.py \
  --repo-id local/ur7e_pick_red_block_singlecam \
  --root ./data/ur7e_pick_red_block_singlecam \
  --task "pick up the red block and place it in the target area" \
  --robot-ip 192.168.1.88 \
  --camera-config record/camera_configs/opencv_single.json \
  --fps 15 \
  --episode-seconds 30 \
  --image-width 320 \
  --image-height 240 \
  --gripper-source do \
  --gripper-index 0
```

### 7.2 双相机采集

```bash
python record/collect_ur7e_multicam_lerobot_v21.py \
  --repo-id local/ur7e_pick_red_block_dualcam \
  --root ./data/ur7e_pick_red_block_dualcam \
  --task "pick up the red block and place it in the target area" \
  --robot-ip 192.168.1.88 \
  --camera-config record/camera_configs/opencv_dual.json \
  --fps 15 \
  --episode-seconds 30 \
  --image-width 320 \
  --image-height 240 \
  --gripper-source do \
  --gripper-index 0
```

### 7.3 三相机采集

```bash
python record/collect_ur7e_multicam_lerobot_v21.py \
  --repo-id local/ur7e_pick_red_block_triplecam \
  --root ./data/ur7e_pick_red_block_triplecam \
  --task "pick up the red block and place it in the target area" \
  --robot-ip 192.168.1.88 \
  --camera-config record/camera_configs/opencv_triple.json \
  --fps 15 \
  --episode-seconds 30 \
  --image-width 320 \
  --image-height 240 \
  --gripper-source do \
  --gripper-index 0
```

### 7.4 RealSense + USB 混合采集

先把模板复制一份，填上真实 RealSense 序列号：

- `record/camera_configs/realsense_dual.template.json`

然后：

```bash
python record/collect_ur7e_multicam_lerobot_v21.py \
  --repo-id local/ur7e_pick_red_block_rs_dual \
  --root ./data/ur7e_pick_red_block_rs_dual \
  --task "pick up the red block and place it in the target area" \
  --robot-ip 192.168.1.88 \
  --camera-config ./my_realsense_dual.json \
  --fps 15 \
  --gripper-source do \
  --gripper-index 0
```

### 7.5 关于 `--fps`

这里的 `--fps` 表示：

- 数据录制节奏
- 数据集元信息里的 fps

不是强制要求所有相机硬件都一定跑这个 fps。

对于普通 USB 相机，脚本默认更保守：

- 使用 `lerobot` 的 OpenCV 相机抽象
- 不强制校验硬件 fps

这在 macOS 上通常更稳。

### 7.6 继续向已有数据集追加

如果你要继续往同一个数据集目录里追加：

```bash
python record/collect_ur7e_multicam_lerobot_v21.py \
  --repo-id local/ur7e_pick_red_block_dualcam \
  --root ./data/ur7e_pick_red_block_dualcam \
  --task "pick up the red block and place it in the target area" \
  --robot-ip 192.168.1.88 \
  --camera-config record/camera_configs/opencv_dual.json \
  --fps 15 \
  --gripper-source do \
  --gripper-index 0 \
  --resume
```

注意：

- `--resume` 时，相机数量、相机名称、分辨率必须和第一次完全一致
- 如果不一致，脚本会直接拒绝

---

## 8. 录制时的交互

脚本启动后会：

1. 连接 UR7e
2. 打开配置文件里声明的相机
3. 测试夹爪读取
4. 提示你把机械臂移动到起始位并开启 Freedrive

录制窗口里可用按键：

- `q` 或 `Esc`
  - 结束当前 episode
- `c`
  - 人工标记夹爪闭合
- `o` 或 `g`
  - 人工标记夹爪打开

episode 结束后，终端会询问：

```text
[s] 保存并继续 / [q] 保存并退出 / [d] 丢弃并继续 / [x] 丢弃并退出
```

---

## 9. 生成的数据集结构

如果本次配置是双相机：

- `front`
- `left_arm`

那么目录大致会是：

```text
data/ur7e_pick_red_block_dualcam/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── meta/
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── info.json
│   └── tasks.jsonl
└── videos/
    └── chunk-000/
        ├── observation.images.front/
        └── observation.images.left_arm/
```

字段固定包含：

- `observation.state`
- `action`

图像字段则由 `camera-config` 决定，例如：

- `observation.images.front`
- `observation.images.left_arm`
- `observation.images.right_arm`

---

## 10. 采集后检查

### 10.1 直接用 `LeRobotDataset` 读取

```bash
conda activate lerobot

python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("lerobot-0.3.3/src").resolve()))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/ur7e_pick_red_block_dualcam",
    root=Path("./data/ur7e_pick_red_block_dualcam").resolve(),
)

print("episodes:", dataset.num_episodes)
print("frames:", dataset.num_frames)
print("camera_keys:", dataset.meta.camera_keys)
print("features:", list(dataset.features))

item = dataset[0]
print("observation.state shape:", item["observation.state"].shape)
print("action shape:", item["action"].shape)
for camera_key in dataset.meta.camera_keys:
    print(camera_key, item[camera_key].shape)
PY
```

这里不要再写死“必须是 3 路相机”，而是看：

- `dataset.meta.camera_keys`

它应该和你这次配置文件里的相机名字一致。

### 10.2 用官方可视化脚本检查

```bash
PYTHONPATH=./lerobot-0.3.3/src python -m lerobot.scripts.visualize_dataset \
  --repo-id local/ur7e_pick_red_block_dualcam \
  --root ./data/ur7e_pick_red_block_dualcam \
  --episode-index 0
```

你可以快速检查：

- 每一路图像有没有录到
- 夹爪标签有没有变化
- 位姿曲线是不是连续

---

## 11. 常见问题

### 11.1 `已有数据集里的特征键与当前相机配置不一致`

原因：

- 你第一次用的是一套相机配置
- `--resume` 时又换成了另一套

处理：

- 同一个数据集目录内，保持同一份 `camera-config`
- 如果要改相机数量或相机键名，新建一个 `--root`

### 11.2 RealSense 扫描或连接失败

优先检查：

- `pyrealsense2` 是否可导入
- 是否真的填了正确的序列号
- USB 线和供电是否稳定

先执行：

```bash
conda activate lerobot
python -c "import pyrealsense2 as rs; print(rs.__version__)"
```

### 11.3 某一路 OpenCV 相机掉帧

优先排查：

- USB 带宽不足
- USB 集线器供电不足
- 分辨率过高

处理建议：

1. 先把 `--fps` 降到 `10` 或 `15`
2. 先把 `--image-width/--image-height` 降到 `320x240`
3. 用 `--sync-mode sequential` 看是否更稳
4. 尽量让多路相机分散到不同 USB 控制器

### 11.4 夹爪状态一直不变

原因通常是：

- 寄存器选错
- 夹爪状态并不在你当前读取的 DO/DI 上

处理：

1. 回头重新跑夹爪诊断
2. 改对 `--gripper-source` 和 `--gripper-index`
3. 不稳定时先切 `--gripper-source manual`

### 11.5 采集结束后编码很慢

这通常是正常的，因为每条 episode 保存时会把临时图像编码成 mp4。

可以优先调小：

- 分辨率
- `--fps`
- `--episode-seconds`

也可以改：

```bash
--video-encoding-batch-size 5
```

---

## 12. 推荐采集顺序

建议按下面顺序走：

1. 先确认 `ur-rtde` 和 `pyrealsense2` 能导入
2. 跑 `--scan-cameras`，确认设备号或序列号
3. 选好一份 `camera-config`
4. 先采一个很小的数据集测试
   - `--num-episodes 1`
   - `--fps 10`
   - `--episode-seconds 10`
5. 用 `LeRobotDataset` 直接读
6. 确认 `camera_keys`、位姿、夹爪都对，再开始批量采集

这样最稳，也最不容易在采了很多条之后才发现相机配置或字段布局错了。

---

## 13. 从 `raw_demos` 转换到 LeRobotDataset v2.1

### 13.1 适用输入

当前仓库里的离线转换命令面向 `XVLA-Code/data_fetch/collect_freedrive.py` 生成的：

- `raw_demos/episode_XXXX.npz`

每个 `npz` 期望包含：

- `images`
- `images_wrist`
- `tcp_poses`
- `gripper`
- `instruction`
- `fps`

其中：

- `images`
  - 主视角 RGB 图像
- `images_wrist`
  - 腕部视角 RGB 图像
- `tcp_poses`
  - `RTDE getActualTCPPose()` 的 6 维轴角位姿
- `gripper`
  - 二值夹爪状态
  - `0.0`
    - 闭合
  - `1.0`
    - 打开

### 13.2 转换命令

推荐直接在 `lerobot` conda 环境中执行：

```bash
source /home/taiyi/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

python record/collect_ur7e_multicam_lerobot_v21.py convert \
  --raw-dir ./raw_demos \
  --root ./data/ur7e_first \
  --repo-id local/ur7e_raw_demos_v21 \
  --main-width 128 \
  --main-height 128 \
  --wrist-width 96 \
  --wrist-height 96
```

说明：

- `--main-width/--main-height`
  - 主视角输出分辨率
- `--wrist-width/--wrist-height`
  - 腕部视角输出分辨率
- 如果不传这些参数
  - 就保持原始分辨率

### 13.3 转换后的字段语义

离线转换和实体机 `record` 模式不完全一样。

这里固定使用 transition 语义：

- `observation.state`
  - 当前时刻 `tcp_x..tcp_rz + gripper`
  - 共 `7` 维
- `action`
  - 下一时刻 `tcp_x..tcp_rz + gripper`
  - 共 `7` 维
- `observation.images.cam_high`
  - 对应原始 `images`
- `observation.images.cam_wrist`
  - 对应原始 `images_wrist`

也就是说：

- 原始一条轨迹有 `T` 帧
- 转换后会生成 `T-1` 条 transition

### 13.4 自动续转规则

`convert` 模式默认支持自动续转，不需要手动写 `--resume`。

规则是：

- `--root` 不存在
  - 新建数据集
- `--root` 已存在且带 `meta/raw_demos_conversion.json`
  - 按 manifest 续转
- `--root` 已存在但没有这个 manifest
  - 直接拒绝，避免脏追加

manifest 会记录：

- 源目录
- 原始分辨率
- 目标分辨率
- fps
- 是否双相机
- 已转换的 `episode_XXXX.npz -> dataset episode index`

如果你第二次运行时修改了这些关键参数，脚本会直接报错而不是悄悄混写。

### 13.5 读取建议

当前这台机器上的 `lerobot` 环境里，`torchcodec` 解码并不稳定。

所以本地读取时建议显式指定：

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/ur7e_raw_demos_v21",
    root=Path("./data/ur7e_raw_demos_v21").resolve(),
    video_backend="pyav",
)
```

### 13.6 校验转换结果

如果你想检查转换后的数据是否“真的符合预期”，推荐直接运行仓库里配套的检查脚本：

- `record/validate_converted_lerobot_v21.py`

示例：

```bash
cd /home/taiyi/project/lerobot_new
source /home/taiyi/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

HF_HOME=/tmp/hf_home_validate \
HF_DATASETS_CACHE=/tmp/hf_datasets_validate \
python record/validate_converted_lerobot_v21.py \
  --root data/ur7e_first \
  --video-backend pyav
```

脚本会自动检查：

- `meta/`、`data/`、`videos/` 目录结构是否完整
- `episodes`、`frames`、`parquet`、`mp4` 数量是否一致
- `raw_demos_conversion.json` 中记录的 fps、分辨率、schema 是否与数据集一致
- `raw_demos` 与转换后数据集的 transition 总数是否一致
- 抽样 episode 的 `observation.state`、`action`、`task` 是否和原始 `npz` 精确对齐
- 抽样图像 shape 是否和 `info.json` 中的目标分辨率一致

正常通过时会看到类似：

```text
[通过] 数据集目录结构完整
[通过] manifest 中的相机分辨率、state/action schema 与数据集一致
[通过] parquet / mp4 文件数与 episode 数量匹配
[通过] raw_demos 与转换后数据集的 transition 总数一致
[通过] 抽样检查了前 3 条 episode 的 state/action/task 数值
[完成] 转换后的 LeRobotDataset v2.1 通过检查
```

如果这里只通过了一部分，通常说明：

- manifest 和当前目录不匹配
- 视频文件数不完整
- 有些原始 `npz` 没有被完整转换
- 转换后 state / action 语义和原始数据没有对齐

---

## 14. 当前 `raw_demos` 轨迹参数记录

以下统计对应仓库当前这批：

- `./raw_demos`

统计时间对应当前仓库状态，具体结果如下：

- episode 数量
  - `50`
- 文件范围
  - `episode_0000.npz` 到 `episode_0064.npz`
- 原始总帧数
  - `29048`
- 可生成 transition 总数
  - `28998`
- 单条轨迹帧数范围
  - 最小 `361`
  - 最大 `900`
  - 平均 `580.96`
- 原始 fps
  - `30`
- 主视角图像分辨率
  - `256 x 256 x 3`
- 腕部视角图像分辨率
  - `256 x 256 x 3`
- 语言指令
  - `Pick up the chili on the table`
- 夹爪取值
  - `0.0`
  - `1.0`

这批轨迹来源于 `XVLA-Code/data_fetch/collect_freedrive.py` 里的当前默认配置：

- `ROBOT_IP`
  - `192.168.1.88`
- `FPS`
  - `30`
- `MAX_FRAMES`
  - `900`
- `TASK_NAME`
  - `Pick up the chili on the table`
- 相机组织
  - 双 RealSense
  - 主视角 + 腕部视角
- 原始保存目录
  - `./raw_demos`

如果你后续又采了新数据，建议同步更新这一节里的：

- episode 数量
- 总帧数 / 总 transitions
- 指令文本
- 原始分辨率
- 推荐转换分辨率

---

## 15. 官方可视化检查

除了上面的自动校验脚本，还可以使用 LeRobot 官方自带的可视化脚本做“肉眼检查”：

- `lerobot.scripts.visualize_dataset`

示例命令：

```bash
cd /home/taiyi/project/lerobot_new
source /home/taiyi/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

PYTHONPATH=./lerobot-0.3.3/src \
python -m lerobot.scripts.visualize_dataset \
  --repo-id local/ur7e_raw_demos_v21 \
  --root ./data/ur7e_first \
  --episode-index 0 \
  --num-workers 0
```

这个官方脚本适合重点观察：

- `observation.images.cam_high`
  - 主视角是否正确
- `observation.images.cam_wrist`
  - 腕部视角是否正确
- `state/*`
  - 位姿曲线是否连续
- `action/*`
  - 下一时刻目标是否和轨迹变化一致

建议实际查看时关注：

- 两路画面是否同步
- 夹爪开合是否和画面一致
- 轨迹中是否存在明显跳变

注意当前环境限制：

- 官方可视化脚本内部不会手动指定 `video_backend=\"pyav\"`
- 如果当前环境中的 `torchcodec` / `ffmpeg` 组合有问题，可能报：
  - `Could not load libtorchcodec`

如果碰到这个错误，优先处理方式是：

1. 修复当前 `lerobot` 环境中的 `torchcodec` 与 `ffmpeg` 兼容性
2. 或在另一个能正常解码 mp4 的 `lerobot` 环境中运行官方脚本

也就是说：

- 自动验结构和数值
  - 用 `record/validate_converted_lerobot_v21.py`
- 肉眼验画面和曲线
  - 用 `python -m lerobot.scripts.visualize_dataset`
