可以参考的内容
1. readme文档：https://github.com/huggingface/lerobot/tree/v0.3.3（请务必学习的是lerobotdatasetv2.1，不要v3的）
2. 一些关于构建代码的说明https://axi404.github.io/blog/lerobot/（还是只是看v2.1的）


在 `lerobot v0.3.3` 中，`LeRobotDataset` 有完整的说明、构造 API 和多视角支持。以下是针对该版本的详细梳理。

---

## 1. 说明文档位置

| 文档类型 | 位置 | 内容 |
|---|---|---|
| **README 格式说明** | 仓库首页 [The `LeRobotDataset` format](https://github.com/huggingface/lerobot/tree/v0.3.3#the-lerobotdataset-format)  | 内部属性结构、`hf_dataset`、`episode_data_index`、`stats`、`info`、`camera_keys` 等 |
| **代码示例** | `examples/1_load_lerobot_dataset.py`  | 完整可运行示例，展示加载、元数据查看、`delta_timestamps`、多帧采样、DataLoader 兼容 |
| **源码 docstring** | `lerobot/common/datasets/lerobot_dataset.py` | `LeRobotDataset` 与 `LeRobotDatasetMetadata` 的类定义与接口 |
| **社区博客/源码解析** | CSDN 等二次文档  | 对 `create`、`add_frame`、`push_to_hub` 等方法的逐步解读 |

> **版本提示**：`v0.3.3` 对应的数据集格式版本为 **v2.1**；`v0.4+` 后升级为 v3.0，结构有差异 。

---

## 2. `LeRobotDataset` 的构造方式

仓库中提供 **两种** 构造路径：加载已有数据集、从零创建新数据集。

### 2.1 加载已有数据集（最常见）

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 从 Hugging Face Hub 加载（自动下载到 ~/.cache/huggingface/lerobot）
dataset = LeRobotDataset("lerobot/aloha_static_coffee")

# 或从本地加载
dataset = LeRobotDataset("lerobot/aloha_static_coffee", root="./my_local_data_dir")

# 只加载指定 episodes
dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23])
```

构造参数（v0.3.3 关键参数）：
- `repo_id`: 数据集 ID
- `root`: 本地根目录（可选）
- `episodes`: 指定加载的 episode 索引列表（可选）
- `image_transforms`: 图像变换函数（可选）
- `delta_timestamps`: 时序采样字典（核心特性，见下文）
- `tolerance_s`: 时间戳对齐容差，默认 `1e-4`
- `download_videos`: 是否下载视频文件

### 2.2 从零创建新数据集（录制/转换）

`LeRobotDataset` 提供一个 **类方法 `create`**，用于初始化空数据集 ：

```python
dataset = LeRobotDataset.create(
    repo_id="your_username/my_dataset",
    fps=30,
    root=None,                    # 本地存储路径
    robot=robot,                  # 机器人对象（自动推断特征）
    robot_type="so100",           # 或手动指定机器人类型
    features={...},               # 手动定义特征（如果不传 robot）
    use_videos=True,              # 是否使用视频编码存储图像
    image_writer_processes=4,     # 图像异步写入进程数
    image_writer_threads=4,       # 图像写入线程数
    video_backend=None,
)
```

创建后通过以下方法填充数据 ：
```python
dataset.add_frame(frame_dict)        # 添加单帧
dataset.add_episode(task="描述")     # 结束当前 episode
dataset.consolidate()                # 最终固化（计算统计量、写入元数据）
dataset.push_to_hub()                # 上传到 Hugging Face Hub
```

---

## 3. 仓库中构建数据集的相关函数/脚本

| 功能 | 位置/函数 | 说明 |
|---|---|---|
| **创建空数据集** | `LeRobotDataset.create()`  | 类方法，初始化目录结构、meta、hf_dataset |
| **添加帧/Episode** | `add_frame()` / `add_episode()`  | 录制数据时使用 |
| **异步图像写入** | `start_image_writer()`  | 多进程/多线程写入视频帧，提升录制效率 |
| **转换已有格式** | `lerobot/scripts/push_dataset_to_hub.py`  | 将 Aloha HDF5、PushT Zarr、XArm PKL 等转换为 LeRobot 格式 |
| **自定义格式转换** | `lerobot/common/datasets/push_dataset_to_hub/${raw_format}_format.py`  | 复制现有模板实现自定义格式 |
| **计算统计量** | `compute_stats.py`  | 计算均值、方差等归一化参数 |
| **可视化检查** | `lerobot/scripts/visualize_dataset.py`  | 用 rerun.io 查看 camera stream、state、action |
| **实体机录制** | `lerobot/scripts/control_robot.py --control.type=record`  | 遥操作录制，内部调用 `LeRobotDataset.create()` |

---

## 4. 多视角（Multi-Camera）支持

**完全支持**。`LeRobotDataset` 的设计目标之一就是处理多模态、多视角的机器人数据 。

### 4.1 存储结构

在 v2.1 格式（v0.3.3）下，每个相机对应独立的视频目录 ：

```
my_dataset/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet   # 包含 observation.images.cam_high 等 VideoFrame 引用
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_high/      # 相机 1
│       │   └── episode_000000.mp4
│       ├── observation.images.cam_wrist/     # 相机 2（手腕视角）
│       │   └── episode_000000.mp4
│       └── observation.images.top/           # 相机 3（俯视）
│           └── episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    └── stats.json
```

Parquet 中不直接存图像像素，而是存 `VideoFrame = {'path': ..., 'timestamp': ...}`，由 `LeRobotDataset` 自动解码并同步 。

### 4.2 代码中使用多视角

**查看有哪些相机**：
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet")
print(ds_meta.camera_keys)
# 输出例如：['observation.images.cam_high', 'observation.images.cam_wrist']
```

**读取特定相机帧**：
```python
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")
frame = dataset[0]

# 多视角图像均以 torch.Tensor (C, H, W) 形式返回
img_high = frame["observation.images.cam_high"]
img_wrist = frame["observation.images.cam_wrist"]
```

**`delta_timestamps` 对多视角分别生效**：
```python
delta_timestamps = {
    "observation.images.cam_high": [-1, -0.5, -0.2, 0],   # 4 帧历史
    "observation.images.cam_wrist": [-1, -0.5, -0.2, 0],  # 4 帧历史
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    "action": [t / dataset.fps for t in range(64)],
}
dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

# 返回的形状：
# dataset[0]["observation.images.cam_high"]  -> (4, C, H, W)
# dataset[0]["observation.images.cam_wrist"] -> (4, C, H, W)
```

### 4.3 创建多视角数据集

在 `create()` 的 `features` 中声明多个 `video` 类型特征即可 ：

```python
dataset = LeRobotDataset.create(
    repo_id="user/my_multi_cam_data",
    fps=30,
    features={
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "observation.images.top": {"dtype": "video", "shape": (480, 640, 3)},
        "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3)},
        "action": {"dtype": "float32", "shape": (6,)},
    },
    use_videos=True,
)
```

---

## 5. 核心特性：`delta_timestamps`

这是 `LeRobotDataset` 区别于普通 PyTorch Dataset 的关键设计 ：

- 普通索引 `dataset[i]` 返回第 `i` 帧；
- 设置 `delta_timestamps` 后，`dataset[i]` 返回相对于第 `i` 帧**时间偏移**的多帧张量。

示例：以 50 FPS 为例，`delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}` 会取当前帧及前 1s、0.5s、0.2s 的帧，返回 `(4, C, H, W)`。

---

## 6. 总结速查

| 问题 | 答案 |
|---|---|
| 是否有说明文档？ | 有，README + `examples/1_load_lerobot_dataset.py` + 源码 docstring |
| 如何构造？ | 加载：`LeRobotDataset(repo_id)`；新建：`LeRobotDataset.create(...)` |
| 是否有构建函数？ | `create()`、`add_frame()`、`add_episode()`、`consolidate()`、`push_to_hub()` |
| 是否支持多视角？ | **完全支持**，每个相机独立视频目录，通过 `camera_keys` 枚举，`features` 中声明多个 `video` 字段 |
| 具体使用文档？ | 见仓库 README 格式章节 、example 脚本 、及 `push_dataset_to_hub.py` 转换文档  |

如果你需要在 `v0.3.3` 上**从零录制一个多视角数据集**，推荐路径是：
1. 参考 `examples/1_load_lerobot_dataset.py` 理解数据格式；
2. 用 `LeRobotDataset.create()` 初始化；
3. 循环调用 `add_frame()` 传入包含多个相机键的字典；
4. 最后 `consolidate()` + `push_to_hub()`。