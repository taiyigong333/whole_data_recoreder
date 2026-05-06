# quality_test

`quality_test/validate_dataset.py` 是这个仓库统一的数据校验入口，支持两种数据格式：

- `XVAL-Code` 风格的 HDF5 数据
- `LeRobot Dataset v2.1`

它的目标是在训练、上传或继续转换之前，尽量提前发现数据问题。

## 校验内容

### 1. 完整性校验

- 必要文件和目录是否存在
- 必要数据集、列、分组是否存在
- episode 文件是否为空
- 视频文件是否存在且非空
- LeRobot 的 `meta/*.json*`、`data/*.parquet`、`videos/*.mp4` 是否彼此一致

### 2. 缺失值校验

- `null`
- `NaN`
- `Inf`
- 空数组
- 不同模态之间帧数不一致

### 3. 数值范围统计

- `joint_position` 的最小值和最大值
- `gripper_position` 的最小值和最大值
- `observation.state` 的逐维最小值和最大值
- `action` 的逐维最小值和最大值
- XVAL HDF5 中 `end_effector` 的逐维最小值和最大值

### 4. 补充的一致性校验

对 `LeRobot v2.1`，还会检查：

- `codebase_version` 是否为 `v2.1`
- `episodes.jsonl`、`episodes_stats.jsonl`、`info.json` 中的计数是否一致
- parquet 行数是否与 episode length 一致
- `frame_index` 是否从 `0` 连续递增
- 全局 `index` 是否连续
- `timestamp` 是否严格递增
- `task_index` 是否都能在 `tasks.jsonl` 中找到
- 如果 `observation.state` 和 `action` 的 schema 相同，校验 `action[t] == observation.state[t+1]`
- 每个应该存在的视频文件是否真的存在

对 `XVAL-Code HDF5`，还会检查：

- `puppet/end_effector` 是否是 `(T, 6)`
- `puppet/joint_position` 是否是 `(T, 7)`
- 图像张量是否是 `(T, H, W, 3)`
- 所有 HDF5 文件的相机 schema 是否一致
- 前 6 个 joint 维度如果全为 0，会给出警告
- `language_instruction` 缺失或为空时给出警告

## 依赖

推荐环境：

```bash
source /home/taiyi/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
python -m pip install -r quality_test/requirements.txt
```

如果你只校验 `LeRobot v2.1`，最关键的依赖是：

- `numpy`
- `pyarrow`

如果你还要校验 `XVAL-Code HDF5`，还需要：

- `h5py`

## 用法

### 自动识别数据格式

```bash
python quality_test/validate_dataset.py data/ur7e_first
```

```bash
python quality_test/validate_dataset.py XVLA-Code/training_data
```

### 强制按 LeRobot v2.1 校验

```bash
python quality_test/validate_dataset.py \
  data/ur7e_first \
  --format lerobot_v21
```

### 强制按 XVAL-Code HDF5 校验

```bash
python quality_test/validate_dataset.py \
  XVLA-Code/training_data \
  --format xval_hdf5
```

### 导出 JSON 报告

```bash
python quality_test/validate_dataset.py \
  data/ur7e_first \
  --json-out quality_test/reports/ur7e_first_report.json
```

### 将 warning 视为失败

```bash
python quality_test/validate_dataset.py \
  data/ur7e_first \
  --strict
```

## 输出说明

脚本会输出：

- 数据概览
- 数值范围统计
- 说明信息
- 警告
- 错误

退出码：

- `0`：校验通过
- `1`：校验失败，或使用了 `--strict` 且存在 warning

## 关于关节范围的说明

这个仓库里有一类 LeRobot 数据集，`observation.state` 保存的是：

- `tcp pose + gripper`

而不是：

- `joint angles + gripper`

这种情况下，校验器不会伪造 joint range，而是会明确提示当前数据集不包含关节位置维度。
