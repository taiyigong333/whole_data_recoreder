#!/usr/bin/env python3
"""UR7e 多机位 LeRobotDataset v2.1 采集入口。

入口脚本保持稳定，具体实现拆分在 `record/ur7e_lerobot_recorder/` 目录：
- `cli.py`：参数解析与运行入口
- `config.py`：相机配置解析与校验
- `convert.py`：raw_demos -> LeRobotDataset v2.1 离线转换
- `recorder.py`：相机、机器人与录制流程
- `dataset.py`：LeRobotDataset 特征与样本转换
"""

from ur7e_lerobot_recorder.cli import main


if __name__ == "__main__":
    main()
