# XVLA-Code

X-VLA 视觉-语言-动作模型部署 — UR7e 机器人抓取项目（代码仓库）

当前仓库里和实体数据最相关的目录有两块：

- `XVLA-Code/data_fetch/`
  - 原始示教数据采集脚本
- `raw_demos/`
  - 已采集好的双视角 `npz` 轨迹

## 目录结构

```
scripts/          推理服务器启动脚本
  start_server_pt.py        X-VLA-Pt 推理服务器
  start_server_libero.py    X-VLA-Libero 推理服务器

evaluation/       评估客户端
  ur5e_client.py            UR5e MuJoCo 仿真评估客户端
  results.json              LIBERO libero_spatial 评估结果 (97.4%)

config/           配置文件
  ur5e_scene.xml            UR5e + Robotiq 2F-85 MuJoCo 仿真场景

models/           模型相关（不放权重文件）
```

## 环境要求

- Python 3.10
- PyTorch + CUDA
- 详见 X-VLA 上游仓库

如果你要把 `raw_demos` 转成 `LeRobotDataset v2.1`，建议直接使用仓库根目录下的：

- `conda` 环境 `lerobot`

---

## 当前 `raw_demos` 参数

当前仓库状态下，仓库根目录 `raw_demos/` 这批轨迹的统计如下：

- episode 数量
  - `50`
- 文件范围
  - `episode_0000.npz` 到 `episode_0064.npz`
- 原始总帧数
  - `29048`
- 可生成 transition 总数
  - `28998`
- 单条轨迹帧数
  - 最小 `361`
  - 最大 `900`
  - 平均 `580.96`
- 原始 fps
  - `30`
- 主视角图像
  - `images`
  - 分辨率 `256 x 256 x 3`
- 腕部视角图像
  - `images_wrist`
  - 分辨率 `256 x 256 x 3`
- 语言指令
  - `Pick up the chili on the table`
- 夹爪取值
  - `0.0` = 闭合
  - `1.0` = 打开

这批数据对应 `data_fetch/collect_freedrive.py` 的当前采集参数：

- `ROBOT_IP=192.168.1.88`
- `FPS=30`
- `MAX_FRAMES=900`
- `TASK_NAME=\"Pick up the chili on the table\"`

---

## 转 LeRobotDataset v2.1

转换入口在仓库根目录：

- `record/collect_ur7e_multicam_lerobot_v21.py convert`

推荐命令：

```bash
cd /home/taiyi/project/lerobot_new
source /home/taiyi/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

python record/collect_ur7e_multicam_lerobot_v21.py convert \
  --raw-dir ./raw_demos \
  --root ./data/ur7e_raw_demos_v21 \
  --repo-id local/ur7e_raw_demos_v21 \
  --main-width 128 \
  --main-height 128 \
  --wrist-width 96 \
  --wrist-height 96
```

转换规则：

- 输入必须是 `episode_*.npz`
- 所有 episode 的 `fps` 必须一致
- 当前这批数据会被识别为双相机数据集
- 每条原始轨迹 `T` 帧会产出 `T-1` 条 LeRobot transition
- `observation.state`
  - 当前时刻 `tcp pose + gripper`
  - 共 `7` 维
- `action`
  - 下一时刻 `tcp pose + gripper`
  - 共 `7` 维

自动续转说明：

- 如果 `--root` 不存在，会新建数据集
- 如果 `--root` 已存在且带 `meta/raw_demos_conversion.json`，会跳过已转换的 episode
- 如果 `--root` 已存在但没有这个 manifest，脚本会拒绝继续，避免脏追加

---

## 读取建议

当前 `lerobot` 环境里 `torchcodec` 解码不稳定，本地读取建议显式指定：

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/ur7e_raw_demos_v21",
    root=Path("./data/ur7e_raw_demos_v21").resolve(),
    video_backend="pyav",
)
```

---

## 检查转换结果

仓库根目录已经提供了一个专门的检查脚本：

- `record/validate_converted_lerobot_v21.py`

推荐命令：

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

这个脚本会检查：

- `meta/`、`data/`、`videos/` 目录结构是否完整
- parquet / mp4 文件数是否和 episode 数一致
- `raw_demos_conversion.json` 中的 fps、分辨率、schema 是否和数据集一致
- `raw_demos` 与转换后数据集的 transition 总数是否一致
- 抽样 episode 的 `observation.state`、`action`、`task` 是否和原始 `npz` 对齐
- 抽样图像张量 shape 是否和 `info.json` 一致

通过时会看到类似：

```text
[通过] 数据集目录结构完整
[通过] manifest 中的相机分辨率、state/action schema 与数据集一致
[通过] raw_demos 与转换后数据集的 transition 总数一致
[通过] 抽样检查了前 3 条 episode 的 state/action/task 数值
[完成] 转换后的 LeRobotDataset v2.1 通过检查
```

---

## 官方可视化

LeRobot 官方自带的可视化脚本是：

- `lerobot.scripts.visualize_dataset`

可以直接查看某一条 episode 的相机画面、state 曲线和 action 曲线：

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

建议优先检查：

- 主视角和腕部视角是否同步、是否录到正确内容
- `state` / `action` 曲线是否连续
- 夹爪开合变化是否和画面一致

注意：

- 这个官方脚本内部不会显式传 `video_backend=\"pyav\"`
- 如果你当前环境里的 `torchcodec` / `ffmpeg` 组合有问题，可能会报 `Could not load libtorchcodec`
- 这种情况下优先：
  - 修复 `ffmpeg` 与 `torchcodec` 兼容性
  - 或换一个能正常 decode 视频的 `lerobot` 环境再运行官方可视化

如果你只是想先判断转换结果是否正确，优先跑上面的校验脚本会更稳。

## 相关仓库

- 上游模型: https://github.com/2toINF/X-VLA
