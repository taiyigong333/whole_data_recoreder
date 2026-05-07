# 自动采集方案说明

当前 `data_fetch_v2` 的自动采集方案做了一个重要调整：

- 示教器端只负责上传 `机械臂位姿 / 机械臂关节角 / 夹爪值`
- 任务描述、PC 端接收频率、保存目录、相机参数、轨迹点、夹爪参数等全部由配置文件统一管理

这样做的好处是：

1. 示教器端脚本更简单，更稳定。
2. 换任务时优先改配置，不需要频繁改 RPC 接口。
3. PC 端和示教器端的职责更清楚。

## 1. 配置文件位置

统一配置文件在：

- `XVLA-Code/data_fetch_v2/configs/auto_collect_config.json`

主要配置项包括：

- `robot.ip`
- `rpc.host`
- `rpc.port`
- `collection.task_name`
- `collection.instruction`
- `collection.sample_hz`
- `paths.save_dir`
- `paths.training_data_dir`
- `cameras`
- `dashboard`
- `ur_script`

## 2. PC 端脚本

PC 端主脚本在：

- `XVLA-Code/data_fetch_v2/fetch_code_auto/auto_collect_rpc.py`

它会完成这些事：

1. 读取配置文件。
2. 启动 XML-RPC 服务。
3. 启动双 RealSense。
4. 在持续收到 `push_sample(...)` 时按配置频率节流抓图。
5. 在 `end_episode(...)` 时保存 `episode_XXXX.npz`。
6. 弹出人工确认，让你决定是否删除该轨迹。
7. 自动生成示教器端脚本 `auto_collect_rpc_program.script`。

## 3. 示教器端 RPC 接口

现在示教器端只用这 3 个 RPC：

1. `begin_episode()`
2. `push_sample(tcp_pose, joint_positions, gripper_position, gripper_closed)`
3. `end_episode(success_flag)`

也就是说，示教器端只传这几类值：

- TCP 位姿 `get_actual_tcp_pose()`
- 关节角 `get_actual_joint_positions()`
- 夹爪位置
- 夹爪闭合标志

## 4. 示教器端脚本来源

示教器端脚本在：

- `XVLA-Code/data_fetch_v2/example_code_in_ur/auto_collect_rpc_program.script`

这个文件不是手工维护为主，而是由 PC 端主脚本根据配置自动生成。

如果你修改了：

- PC 端接收频率
- 夹爪参数
- 轨迹点
- RPC 地址

重新启动一次 `auto_collect_rpc.py`，它会把示教器脚本重新生成出来。

## 5. 运行流程

### 第一步：修改配置

先按你的现场情况修改：

- `XVLA-Code/data_fetch_v2/configs/auto_collect_config.json`

尤其要看：

- `robot.ip`
- `collection`
- `paths`
- `cameras`
- `ur_script.gripper_prefix`
- `ur_script.home_q`
- `ur_script.pick_pre`
- `ur_script.pick_pose`
- `ur_script.place_pre`
- `ur_script.place_pose`

### 第二步：启动 PC 端服务

```powershell
conda activate XVLA
cd F:\research\vla_team\XVLA-Code\data_fetch_v2\fetch_code_auto
python auto_collect_rpc.py
```

如果本次想临时跳过相机预览：

```powershell
python auto_collect_rpc.py --no-camera-preview
```

### 第三步：把自动生成的脚本贴到示教器

启动 PC 脚本后，会自动更新：

- `XVLA-Code/data_fetch_v2/example_code_in_ur/auto_collect_rpc_program.script`

把这份脚本粘贴到示教器的 Script 节点里即可。

### 第四步：执行采集

示教器端运行脚本后：

1. 调用 `begin_episode()`
2. 循环发送样本
3. 结束后调用 `end_episode(1)`

PC 端会保存数据，并询问你是否保留。

## 6. 数据保存内容

每条轨迹保存为 `episode_XXXX.npz`，字段包括：

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

## 7. 转换为 HDF5

转换脚本在：

- `XVLA-Code/data_fetch_v2/fetch_code_auto/convert_auto_demos_to_hdf5.py`

默认也会读取同一个配置文件：

```powershell
conda activate XVLA
cd F:\research\vla_team\XVLA-Code\data_fetch_v2\fetch_code_auto
python convert_auto_demos_to_hdf5.py
```

如果你想覆盖输入输出目录，也可以手动指定：

```powershell
python convert_auto_demos_to_hdf5.py --input-dir .\raw_demos --output-dir .\training_data
```

## 8. 现阶段假设

当前实现默认以下条件成立：

1. PC 上有两台可用 RealSense。
2. 机器人能访问到 PC 的 XML-RPC 地址。
3. 示教器端支持 `rpc_factory("xmlrpc", ...)`。
4. DH AG 的 URCap 已正确安装。
5. 你的 DH URCap 前缀和配置中填写的一致。

## 9. 如果后面还要继续增强

后面最值得继续补的方向有两个：

1. 把当前示例轨迹扩展成多任务模板。
2. 如果确认 URCap 支持真实夹爪反馈值，把当前“命令位置”改成“真实位置反馈”。
