# `data_fetch_v2` 使用说明

`data_fetch_v2` 用于完成 `UR7e + DH AG + 双 RealSense` 的自动化数据采集，并支持在每条轨迹采集完成后手动确认是否删除。

这一版的核心设计是：

- 示教器端只传递机械臂和夹爪的值
- 任务参数、PC 端接收频率、保存路径、相机参数、轨迹点等统一由配置文件管理

## 目录结构

```text
data_fetch_v2/
├─ README.md
├─ configs/
│  └─ auto_collect_config.json
├─ fetch_code_auto/
│  ├─ auto_collect_rpc.py
│  ├─ config_utils.py
│  └─ convert_auto_demos_to_hdf5.py
└─ example_code_in_ur/
   ├─ auto_collect_rpc_program.script
   ├─ auto_collect_rpc_program.md
   └─ sample_gripper_xml.md
```

## 方案概览

整体流程如下：

1. PC 端读取配置文件。
2. PC 端启动 XML-RPC 服务。
3. PC 端启动双 RealSense。
4. PC 端根据配置自动生成示教器脚本。
5. 示教器端运行脚本，只上传机械臂和夹爪状态。
6. PC 端按配置频率从持续上报的状态中节流抓图。
7. 轨迹结束后，PC 保存数据并询问是否保留。

## 配置文件

统一配置文件在 [auto_collect_config.json](./configs/auto_collect_config.json)。

主要字段如下：

`robot`

- `ip`：UR7e 的 IP 地址。

`rpc`

- `host`：PC 端 XML-RPC 绑定地址。
- `port`：PC 端 XML-RPC 端口。

`collection`

- `task_name`：任务名。
- `instruction`：语言指令。
- `sample_hz`：PC 端接收并保存样本的频率。
- `expected_samples`：期望样本数，可选。
- `operator_note`：保存到元数据的备注。
- `min_frames`：低于这个帧数时默认不保留。

`paths`

- `save_dir`：原始 `.npz` 轨迹保存目录。
- `training_data_dir`：转换后的 `.hdf5` 保存目录。

`cameras`

- `fps`：采图帧率。
- `width` / `height`：RealSense 原始分辨率。
- `output_width` / `output_height`：保存时 resize 后尺寸。
- `preview_enabled`：是否启用相机顺序确认预览。

`dashboard`

- `auto_start_program`：是否通过 Dashboard 自动加载 UR 程序。
- `program_name`：要加载的 UR 程序名。

`ur_script`

- `rpc_url`：示教器端 XML-RPC 地址，填 `auto` 时自动生成。
- `sample_hz`：示教器端发送频率配置。当前示例脚本使用持续发送 + `sync()`，该字段可保留为 `0.0`。
- `gripper_prefix`：DH URCap 函数前缀。
- `gripper_index`：夹爪编号。
- `gripper_force` / `gripper_speed`
- `gripper_open_position` / `gripper_close_position`
- `home_q`
- `pick_pre`
- `pick_pose`
- `place_pre`
- `place_pose`

## PC 端采集脚本

主脚本是 [auto_collect_rpc.py](./fetch_code_auto/auto_collect_rpc.py)。

它会完成这些事情：

1. 读取配置文件。
2. 启动 XML-RPC 服务。
3. 打开两台 RealSense。
4. 自动生成示教器端脚本。
5. 响应 `begin_episode / push_sample / end_episode`。
6. 将图像与机器人状态对齐保存。
7. 在每条轨迹结束后允许手动决定是否删除。

### 启动方式

如果当前目录在 `data_fetch_v2` 下：

```powershell
conda activate XVLA
cd .\fetch_code_auto
python auto_collect_rpc.py
```

如果想临时关闭相机预览：

```powershell
python auto_collect_rpc.py --no-camera-preview
```

如果要指定配置文件：

```powershell
python auto_collect_rpc.py --config ..\configs\auto_collect_config.json
```

## 示教器端脚本

自动生成的示教器脚本是 [auto_collect_rpc_program.script](./example_code_in_ur/auto_collect_rpc_program.script)。

相关说明文件：

- [auto_collect_rpc_program.md](./example_code_in_ur/auto_collect_rpc_program.md)
- [sample_gripper_xml.md](./example_code_in_ur/sample_gripper_xml.md)

### 示教器端只负责什么

示教器端现在只负责：

1. 调用 `begin_episode()`
2. 持续调用 `push_sample(...)`
3. 调用 `end_episode(1)`

上传的值只有：

- `get_actual_tcp_pose()`
- `get_actual_joint_positions()`
- `current_gripper_pos`
- `gripper_closed_flag()`

### 为什么这样设计

这样可以把更容易变化的内容集中放进配置文件，例如：

- 任务语言
- PC 端接收频率
- 轨迹点
- 夹爪参数
- 保存路径

## 数据格式

每条轨迹保存为一个 `episode_XXXX.npz` 文件，典型字段如下：

- `images`
- `images_wrist`
- `tcp_poses`
- `joint_positions`
- `gripper`
- `gripper_position`
- `gripper_closed`
- `robot_timestamps`
- `pc_timestamps`
- `sample_indices`
- `instruction`
- `task_name`
- `metadata_json`

其中：

- `gripper` 是归一化开合比例，范围 `[0, 1]`
- `gripper_position` 是原始夹爪位置，范围 `[0, 100]`

## 采集完成后的删除确认

每条轨迹结束后，PC 会：

1. 生成预览图。
2. 显示首帧、中间帧、末帧。
3. 询问是否保留。

如果轨迹帧数低于 `collection.min_frames`，默认选项会偏向删除。

## 转换为 HDF5

转换脚本是 [convert_auto_demos_to_hdf5.py](./fetch_code_auto/convert_auto_demos_to_hdf5.py)。

### 默认转换

如果当前目录在 `data_fetch_v2` 下：

```powershell
conda activate XVLA
cd .\fetch_code_auto
python convert_auto_demos_to_hdf5.py
```

### 指定输入输出目录

```powershell
python convert_auto_demos_to_hdf5.py --input-dir .\raw_demos --output-dir .\training_data
```

### 转换结果

它会把：

- `images`
- `images_wrist`
- `tcp_poses`
- `joint_positions`
- `gripper`

转换成训练用的 HDF5 结构：

- `observations/images/cam_high`
- `observations/images/cam_wrist`
- `puppet/end_effector`
- `puppet/joint_position`

## 推荐使用顺序

1. 先修改 [auto_collect_config.json](./configs/auto_collect_config.json)。
2. 运行 [auto_collect_rpc.py](./fetch_code_auto/auto_collect_rpc.py)。
3. 打开自动生成的 [auto_collect_rpc_program.script](./example_code_in_ur/auto_collect_rpc_program.script)。
4. 贴到示教器的 Script 节点。
5. 运行采集。
6. 在 PC 端确认保留或删除。
7. 采集完成后运行 [convert_auto_demos_to_hdf5.py](./fetch_code_auto/convert_auto_demos_to_hdf5.py)。

## 注意事项

1. 当前夹爪上报值默认是“命令位置”，不是“URCap 真实反馈位置”。
2. 如果你确认自己的 URCap 支持真实位置读回，可以继续替换成真实反馈值。
3. 如果 DH URCap 的函数前缀不是 `dh_ag95`，请修改配置中的 `ur_script.gripper_prefix`。
4. 如果你修改了配置，建议重新启动一次 [auto_collect_rpc.py](./fetch_code_auto/auto_collect_rpc.py)，这样会重新生成示教器脚本。
