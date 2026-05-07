from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


# data_fetch_v2 根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认配置文件位置
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "auto_collect_config.json"


# 默认配置。
# 真实运行时会先加载这份默认值，再用用户配置文件中的内容覆盖。
DEFAULT_CONFIG: Dict[str, Any] = {
    "robot": {
        "ip": "192.168.1.88",
    },
    "rpc": {
        "host": "0.0.0.0",
        "port": 50000,
    },
    "collection": {
        "task_name": "ur7e_dh_ag_demo",
        "instruction": "Pick up the chili on the table",
        "sample_hz": 30.0,
        "expected_samples": 0,
        "operator_note": "",
        "min_frames": 20,
    },
    "paths": {
        "save_dir": "./fetch_code_auto/raw_demos",
        "training_data_dir": "./fetch_code_auto/training_data",
    },
    "cameras": {
        "fps": 30,
        "width": 640,
        "height": 480,
        "output_width": 256,
        "output_height": 256,
        "preview_enabled": True,
        "live_view_width": 512,
        "live_view_height": 512,
    },
    "dashboard": {
        "auto_start_program": False,
        "program_name": "auto_collect_rpc.urp",
    },
    "ur_script": {
        "rpc_url": "auto",
        "sample_hz": 0.0,
        "gripper_prefix": "dh_ag95",
        "gripper_index": 1,
        "activate_gripper": True,
        "gripper_force": 40,
        "gripper_speed": 40,
        "gripper_open_position": 100.0,
        "gripper_close_position": 0.0,
        "gripper_closed_threshold": 50.0,
        "wait_after_open_s": 0.8,
        "wait_after_close_s": 1.0,
        "home_q": [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0],
        "pick_pre": [0.45, -0.20, 0.25, 2.22, -2.22, 0.0],
        "pick_pose": [0.45, -0.20, 0.12, 2.22, -2.22, 0.0],
        "place_pre": [0.30, 0.25, 0.25, 2.22, -2.22, 0.0],
        "place_pose": [0.30, 0.25, 0.12, 2.22, -2.22, 0.0],
        "home_acc": 1.2,
        "home_vel": 1.0,
        "linear_acc": 1.2,
        "linear_fast_vel": 0.25,
        "linear_slow_vel": 0.20,
    },
}


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | Path | None = None) -> tuple[Dict[str, Any], Path]:
    """加载配置文件，并与默认配置合并。"""
    if config_path is None:
        config_file = DEFAULT_CONFIG_PATH
    else:
        config_file = Path(config_path).resolve()

    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_file.exists():
        user_config = json.loads(config_file.read_text(encoding="utf-8"))
        config = deep_merge_dict(config, user_config)

    return config, config_file


def resolve_project_path(path_str: str | Path) -> Path:
    """把相对路径解析为相对于 data_fetch_v2 根目录的绝对路径。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()
