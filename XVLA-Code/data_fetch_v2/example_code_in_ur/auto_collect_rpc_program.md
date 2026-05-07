# 示教器端脚本说明

这个文件由 `fetch_code_auto/auto_collect_rpc.py` 根据配置自动生成。

当前脚本是一个最小化的示教器模板：

- 适合直接放在示教器 `Script` 节点
- 不使用 `def`
- 只上传三类值：

- `current_gripper_pos`：夹爪角度 / 开合量
- `get_actual_tcp_pose()`：TCP 六维绝对位姿
- `get_actual_joint_positions()`：六个关节相对角度

当前 XML-RPC 地址：

```text
http://192.168.1.100:50000/RPC2
```

示教器端实际调用流程：

1. `begin_episode()`
2. 持续 `push_sample(tcp_pose, joint_positions, gripper_position)`
3. `end_episode(1)`

示教器端上传的只有：

- TCP 六维绝对位姿
- 六个关节相对角度
- 夹爪角度 / 开合量

如果你在示教器程序的其他位置控制夹爪，请同步更新：

`current_gripper_pos`

这样发送给 PC 的夹爪值才会和实际指令一致。

如果你要修改 PC 端接收频率、夹爪参数或 RPC 地址，请直接编辑：

`XVLA-Code/data_fetch_v2/configs/auto_collect_config.json`
