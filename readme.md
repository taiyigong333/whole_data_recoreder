# UR7e 数据采集、转换与校验

这个仓库主要包含 4 条能力链路：

- 采集 UR7e 示教数据
- 将原始数据转换为 `LeRobot Dataset v2.1`
- 将原始数据导出为 `XVAL-Code` 风格的 HDF5
- 在训练前做统一的数据质量校验

## 主要目录

### `record/`

UR7e 的采集、转换和 LeRobot 相关工具。

关键入口：

- `record/collect_ur7e_multicam_lerobot_v21.py`
- `record/validate_converted_lerobot_v21.py`

适合用在：

- 多相机采集
- 将 `raw_demos/*.npz` 转为 `LeRobot Dataset v2.1`
- 检查转换后的 LeRobot 数据集是否正确

### `XVLA-Code/`

UR7e 原始数据采集脚本，以及 XVAL 训练相关的辅助脚本。

关键入口：

- `XVLA-Code/data_fetch/collect_freedrive.py`
- `XVLA-Code/data_fetch/convert_to_hdf5.py`

适合用在：

- 采集原始 `npz` 轨迹
- 将原始数据转为 `XVAL-Code` 风格的 HDF5

### `quality_test/`

统一的数据校验工具，支持：

- `LeRobot Dataset v2.1`
- `XVAL-Code` HDF5 数据

关键入口：

- `quality_test/validate_dataset.py`
- `quality_test/README.md`

## 推荐工作流

### 1. 采集原始数据

如果你要采集 XVAL 风格的原始示教数据，优先使用 `XVLA-Code/data_fetch/` 下的脚本。

如果你要直接走 LeRobot 的采集或转换流程，优先使用 `record/`。

### 2. 转换数据

将 `npz` 转为 LeRobot：

```bash
python record/collect_ur7e_multicam_lerobot_v21.py convert \
  --raw-dir ./raw_demos \
  --root ./data/ur7e_first \
  --repo-id local/ur7e_first
```

这条离线转换固定按 `TCP + gripper` 写入 `LeRobot Dataset v2.1`：

- `observation.state = tcp[t] + gripper[t]`
- `action = tcp[t+1] + gripper[t+1]`

不是关节角数据。

将 `npz` 转为 XVAL HDF5：

```bash
python XVLA-Code/data_fetch/convert_to_hdf5.py
```

### 3. 训练前做数据校验

校验 LeRobot v2.1：

```bash
python quality_test/validate_dataset.py data/ur7e_first
```

要核对 `raw_demos -> LeRobot v2.1` 的 `TCP` 和夹爪是否一一对齐，建议再跑：

```bash
HF_HOME=/tmp/hf_home_validate \
HF_DATASETS_CACHE=/tmp/hf_datasets_validate \
python record/validate_converted_lerobot_v21.py \
  --root data/ur7e_first \
  --raw-dir ./raw_demos \
  --video-backend pyav
```

它会直接展示：

- 原始和转换后的 `TCP`
- 原始和转换后的夹爪值
- 每个抽样 episode 的最大绝对误差

校验 XVAL HDF5：

```bash
python quality_test/validate_dataset.py XVLA-Code/training_data --format xval_hdf5
```

## `quality_test` 增加了什么

相比之前偏一次性的检查脚本，`quality_test` 更偏可复用、可扩展的质量校验：

- 缺文件、缺列、缺数据集检测
- `null`、`NaN`、`Inf` 检测
- 逐维最小值和最大值统计
- 夹爪范围统计
- 当数据集中真的包含 joint 维度时，输出 joint 范围
- LeRobot 的 meta、parquet、video 一致性校验
- XVAL HDF5 的相机 schema 一致性校验
- LeRobot transition 连续性校验

## 环境

推荐基础环境：

```bash
source /home/taiyi/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
```

安装 `quality_test` 依赖：

```bash
python -m pip install -r quality_test/requirements.txt
```

## 备注

- LeRobot 这条链路请保持在 `LeRobot Dataset v2.1`
- 这个仓库中的 `raw_demos -> LeRobot` 离线转换保存的是 `tcp pose + gripper`，不是完整 joint angle
- `record/validate_converted_lerobot_v21.py` 会直接比对转换前后的 `TCP` 和夹爪值，并输出结果摘要
