from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Iterable, List, Sequence
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

import cv2
import numpy as np
import pyrealsense2 as rs

from config_utils import DEFAULT_CONFIG_PATH, load_config, resolve_project_path


class RPCRequestHandler(SimpleXMLRPCRequestHandler):
    """限制 XML-RPC 可访问的路径。"""

    rpc_paths = ("/", "/RPC2")


class CompatibleXMLRPCServer(SimpleXMLRPCServer):
    """允许快速重启服务端口。"""

    allow_reuse_address = True


class DashboardClient:
    """用于通过 Dashboard 端口加载并启动示教器程序。"""

    def __init__(self, robot_ip: str, port: int = 29999) -> None:
        self.robot_ip = robot_ip
        self.port = port

    def send(self, command: str) -> str:
        with socket.create_connection((self.robot_ip, self.port), timeout=3.0) as sock:
            sock.settimeout(2.0)
            try:
                sock.recv(1024)
            except socket.timeout:
                pass
            if not command.endswith("\n"):
                command += "\n"
            sock.sendall(command.encode("utf-8"))
            try:
                return sock.recv(1024).decode("utf-8", errors="ignore").strip()
            except socket.timeout:
                return ""

    def load_and_play(self, program_name: str) -> None:
        reply = self.send(f"load {program_name}")
        if reply:
            print(f"[dashboard] load -> {reply}")
        time.sleep(1.0)
        reply = self.send("play")
        if reply:
            print(f"[dashboard] play -> {reply}")


def clamp(value: float, low: float, high: float) -> float:
    """把数值限制在给定区间内。"""
    return max(low, min(high, value))


def guess_local_ip(robot_ip: str) -> str:
    """根据机器人 IP 猜测本机对外通信所用的 IP。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((robot_ip, 1))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def coerce_float_list(values: Iterable[float], expected_len: int, name: str) -> List[float]:
    """把 RPC 传来的列表规范化成指定长度的 float 列表。"""
    values = list(values)
    if len(values) < expected_len:
        raise ValueError(f"{name} 至少需要 {expected_len} 个值，实际收到 {len(values)} 个")
    return [float(v) for v in values[:expected_len]]


def normalize_gripper_position(raw_position: float) -> tuple[float, float]:
    """统一夹爪值表达。

    支持两种输入：
    - [0, 100]：直接视作原始位置
    - [0, 1]：自动放缩到 [0, 100]
    """
    position = float(raw_position)
    if 0.0 <= position <= 1.0:
        position = position * 100.0
    position = clamp(position, 0.0, 100.0)
    open_ratio = position / 100.0
    return position, open_ratio


def format_pose_as_urscript(values: Sequence[float]) -> str:
    """把 6 维位姿格式化为 URScript 的 p[...] 文本。"""
    return "p[" + ", ".join(f"{float(v):.6f}" for v in values[:6]) + "]"


def format_joint_as_urscript(values: Sequence[float]) -> str:
    """把关节角列表格式化为 URScript 的 [...] 文本。"""
    return "[" + ", ".join(f"{float(v):.6f}" for v in values[:6]) + "]"


class EpisodeBuffer:
    """缓存一条轨迹在保存前的所有样本。"""

    def __init__(
        self,
        episode_index: int,
        task_name: str,
        instruction: str,
        sample_hz: float,
        expected_samples: int,
        operator_note: str,
    ) -> None:
        self.episode_index = episode_index
        self.task_name = task_name
        self.instruction = instruction
        self.sample_hz = sample_hz
        self.expected_samples = expected_samples
        self.operator_note = operator_note
        self.created_at = time.time()
        self.last_accept_time_s: float | None = None
        self.images_main: List[np.ndarray] = []
        self.images_wrist: List[np.ndarray] = []
        self.tcp_poses: List[List[float]] = []
        self.joint_positions: List[List[float]] = []
        self.gripper_ratio: List[float] = []
        self.gripper_position: List[float] = []
        self.gripper_closed: List[float] = []
        self.robot_timestamps: List[float] = []
        self.pc_timestamps: List[float] = []
        self.sample_indices: List[int] = []

    @property
    def sample_count(self) -> int:
        return len(self.images_main)


class AutoCollectRPCService:
    """PC 端自动采集服务。

    设计原则：
    - 示教器只负责上传机械臂和夹爪的值
    - 任务信息、路径、相机参数、采样频率都走 config 管理
    """

    def __init__(self, runtime_config: dict, config_path: Path) -> None:
        self.runtime_config = runtime_config
        self.config_path = config_path

        collection_cfg = runtime_config["collection"]
        path_cfg = runtime_config["paths"]
        camera_cfg = runtime_config["cameras"]

        self.robot_ip = runtime_config["robot"]["ip"]
        self.task_name = collection_cfg["task_name"]
        self.instruction = collection_cfg["instruction"]
        self.sample_hz = float(collection_cfg["sample_hz"])
        if self.sample_hz <= 0.0:
            raise ValueError("collection.sample_hz 必须大于 0")
        self.sample_period_s = 1.0 / self.sample_hz
        self.expected_samples = int(collection_cfg["expected_samples"])
        self.operator_note = str(collection_cfg["operator_note"])
        self.min_frames = int(collection_cfg["min_frames"])

        self.camera_fps = int(camera_cfg["fps"])
        if self.camera_fps <= 0:
            raise ValueError("cameras.fps 必须大于 0")
        self.camera_width = int(camera_cfg["width"])
        self.camera_height = int(camera_cfg["height"])
        self.output_width = int(camera_cfg["output_width"])
        self.output_height = int(camera_cfg["output_height"])
        self.preview_enabled = bool(camera_cfg["preview_enabled"])
        self.live_view_width = int(camera_cfg["live_view_width"])
        self.live_view_height = int(camera_cfg["live_view_height"])

        self.save_dir = resolve_project_path(path_cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir = self.save_dir / "preview"
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        self.episode_index = self._find_next_episode_index()
        self.current_episode: EpisodeBuffer | None = None

        self.pipelines = self._start_realsense_pipelines()
        self.active_pipelines = self._confirm_camera_order()

    def _find_next_episode_index(self) -> int:
        """从已有数据中推断下一个轨迹编号。"""
        max_index = -1
        for path in self.save_dir.glob("episode_*.npz"):
            try:
                max_index = max(max_index, int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return max_index + 1

    def _start_realsense_pipelines(self) -> list[dict]:
        """启动双 RealSense 相机。"""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) < 2:
            raise RuntimeError(f"至少需要 2 台 RealSense，相机实际数量为 {len(devices)}")

        pipelines = []
        print(f"[camera] 检测到 {len(devices)} 台 RealSense")
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(
                rs.stream.color,
                self.camera_width,
                self.camera_height,
                rs.format.bgr8,
                self.camera_fps,
            )
            pipeline.start(config)
            pipelines.append(
                {
                    "name": name,
                    "serial": serial,
                    "pipe": pipeline,
                }
            )
            print(f"[camera] 已启动 {name} (S/N: {serial})")

        return pipelines[:2]

    def _confirm_camera_order(self) -> list[dict]:
        """人工确认主视角和腕部视角的对应关系。"""
        main_cam = self.pipelines[0]
        wrist_cam = self.pipelines[1]

        if not self.preview_enabled:
            print("[camera] 已跳过相机预览，默认第一台为主视角")
            return [main_cam, wrist_cam]

        print("[camera] 按 Enter 确认相机顺序，按 S 交换左右视角")
        while True:
            try:
                frame_main = main_cam["pipe"].wait_for_frames(timeout_ms=2000).get_color_frame()
                frame_wrist = wrist_cam["pipe"].wait_for_frames(timeout_ms=2000).get_color_frame()
                if frame_main and frame_wrist:
                    img_main = np.asanyarray(frame_main.get_data())
                    img_wrist = np.asanyarray(frame_wrist.get_data())
                    cv2.putText(
                        img_main,
                        f"Main: {main_cam['name']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        img_wrist,
                        f"Wrist: {wrist_cam['name']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                    preview = np.hstack([img_main, img_wrist])
                    cv2.imshow("Camera Preview", preview)
            except RuntimeError:
                pass

            key = cv2.waitKey(100) & 0xFF
            if key == 13:
                break
            if key == ord("s"):
                main_cam, wrist_cam = wrist_cam, main_cam
                print(f"[camera] 已交换 -> 主视角={main_cam['name']} 腕部={wrist_cam['name']}")

        cv2.destroyWindow("Camera Preview")
        return [main_cam, wrist_cam]

    def close(self) -> None:
        """释放相机和窗口资源。"""
        for item in self.active_pipelines:
            try:
                item["pipe"].stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

    def _capture_rgb_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """同步抓取主视角和腕部视角 RGB 图像。"""
        frames = []
        for item in self.active_pipelines:
            bundle = item["pipe"].wait_for_frames(timeout_ms=2000)
            color_frame = bundle.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"相机 {item['name']} 没有返回彩色图像")
            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (self.output_width, self.output_height)))
        return frames[0], frames[1]

    def _show_live_preview(
        self,
        image_main: np.ndarray,
        image_wrist: np.ndarray,
        sample_index: int,
        gripper_position: float,
    ) -> None:
        """显示采集中的实时预览。"""
        main_bgr = cv2.cvtColor(image_main, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(image_wrist, cv2.COLOR_RGB2BGR)
        main_bgr = cv2.resize(main_bgr, (self.live_view_width, self.live_view_height))
        wrist_bgr = cv2.resize(wrist_bgr, (self.live_view_width, self.live_view_height))
        cv2.putText(
            main_bgr,
            f"episode={self.current_episode.episode_index:04d} sample={sample_index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            main_bgr,
            f"gripper={gripper_position:.1f}",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            wrist_bgr,
            "wrist",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow("XVLA Auto Collection", np.hstack([main_bgr, wrist_bgr]))
        cv2.waitKey(1)

    def _build_review_sheet(self, episode: EpisodeBuffer) -> np.ndarray:
        """生成轨迹保存后的复核预览图。"""
        indices = [0, max(0, episode.sample_count // 2), max(0, episode.sample_count - 1)]

        def build_row(images: List[np.ndarray], label: str) -> np.ndarray:
            panels = []
            for index in indices:
                image = cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (320, 240))
                cv2.putText(
                    image,
                    f"{label} frame {index}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                panels.append(image)
            return np.hstack(panels)

        main_row = build_row(episode.images_main, "main")
        wrist_row = build_row(episode.images_wrist, "wrist")

        footer = np.zeros((120, main_row.shape[1], 3), dtype=np.uint8)
        info_lines = [
            f"episode_{episode.episode_index:04d}  samples={episode.sample_count}  hz={episode.sample_hz:.1f}",
            f"task={episode.task_name}",
            f"instruction={episode.instruction}",
        ]
        if episode.operator_note:
            info_lines.append(f"note={episode.operator_note}")

        y = 28
        for line in info_lines:
            cv2.putText(
                footer,
                line[:110],
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            y += 30

        return np.vstack([main_row, wrist_row, footer])

    def _write_episode(self, episode: EpisodeBuffer) -> tuple[Path, Path]:
        """把缓存中的一条轨迹保存为 npz，并写出预览图。"""
        save_path = self.save_dir / f"episode_{episode.episode_index:04d}.npz"
        preview_path = self.preview_dir / f"episode_{episode.episode_index:04d}.jpg"

        metadata = {
            "task_name": episode.task_name,
            "instruction": episode.instruction,
            "sample_hz": episode.sample_hz,
            "expected_samples": episode.expected_samples,
            "operator_note": episode.operator_note,
            "created_at": episode.created_at,
            "robot_ip": self.robot_ip,
            "camera_fps": self.camera_fps,
            "config_path": str(self.config_path),
        }

        np.savez_compressed(
            save_path,
            images=np.asarray(episode.images_main, dtype=np.uint8),
            images_wrist=np.asarray(episode.images_wrist, dtype=np.uint8),
            tcp_poses=np.asarray(episode.tcp_poses, dtype=np.float64),
            joint_positions=np.asarray(episode.joint_positions, dtype=np.float64),
            gripper=np.asarray(episode.gripper_ratio, dtype=np.float32),
            gripper_position=np.asarray(episode.gripper_position, dtype=np.float32),
            gripper_closed=np.asarray(episode.gripper_closed, dtype=np.float32),
            robot_timestamps=np.asarray(episode.robot_timestamps, dtype=np.float64),
            pc_timestamps=np.asarray(episode.pc_timestamps, dtype=np.float64),
            sample_indices=np.asarray(episode.sample_indices, dtype=np.int32),
            instruction=episode.instruction,
            task_name=episode.task_name,
            fps=np.asarray(episode.sample_hz, dtype=np.float32),
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )

        review_sheet = self._build_review_sheet(episode)
        cv2.imwrite(str(preview_path), review_sheet)
        cv2.imshow("Episode Review", review_sheet)
        cv2.waitKey(1)
        return save_path, preview_path

    def _prompt_keep_episode(self, save_path: Path, sample_count: int) -> bool:
        """保存结束后询问用户是否保留该轨迹。"""
        if sample_count < self.min_frames:
            prompt = f"{save_path.name} 只有 {sample_count} 帧，仍然保留吗？[y/N]: "
            keep_default = False
        else:
            prompt = f"保留 {save_path.name} 吗？[Y/n]: "
            keep_default = True

        while True:
            answer = input(prompt).strip().lower()
            if answer == "":
                return keep_default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no", "d", "delete"}:
                return False
            print("请输入 y/yes 或 n/no。")

    def ping(self) -> str:
        """连通性测试接口。"""
        return "pong"

    def begin_episode(self) -> int:
        """开始采集一条新轨迹。

        示教器端无需再传 task/instruction/sample_hz，
        这些参数全部从配置文件读取。
        """
        if self.current_episode is not None:
            print("[rpc] begin_episode 被拒绝：当前已有未结束的轨迹")
            return -1

        self.current_episode = EpisodeBuffer(
            episode_index=self.episode_index,
            task_name=self.task_name,
            instruction=self.instruction,
            sample_hz=self.sample_hz,
            expected_samples=self.expected_samples,
            operator_note=self.operator_note,
        )
        print(
            "[rpc] begin_episode -> "
            f"episode_{self.current_episode.episode_index:04d} "
            f"task={self.current_episode.task_name} hz={self.current_episode.sample_hz:.1f}"
        )
        return self.current_episode.episode_index

    def push_sample(
        self,
        tcp_pose: Sequence[float],
        joint_positions: Sequence[float],
        gripper_position: float,
        gripper_closed: float = 0.0,
    ) -> int:
        """追加一个采样点。

        示教器端只上传：
        - 机械臂 TCP 位姿
        - 机械臂关节角
        - 夹爪位置
        - 夹爪开闭标志（可选）
        """
        if self.current_episode is None:
            print("[rpc] push_sample 被忽略：当前没有活动轨迹")
            return -1

        now = time.time()
        episode = self.current_episode
        if episode.last_accept_time_s is not None:
            elapsed_s = now - episode.last_accept_time_s
            # PC 端按配置节流保存频率；示教器端可以持续发送。
            if elapsed_s + 1e-9 < self.sample_period_s:
                return episode.sample_count

        try:
            tcp = coerce_float_list(tcp_pose, 6, "tcp_pose")
            joints = coerce_float_list(joint_positions, 6, "joint_positions")
            raw_gripper, open_ratio = normalize_gripper_position(float(gripper_position))
            image_main, image_wrist = self._capture_rgb_pair()
        except Exception as exc:
            print(f"[rpc] push_sample 失败: {exc}")
            return -2

        sample_index = episode.sample_count
        robot_time_s = sample_index / self.sample_hz
        pc_time_s = now

        episode.images_main.append(image_main)
        episode.images_wrist.append(image_wrist)
        episode.tcp_poses.append(tcp)
        episode.joint_positions.append(joints)
        episode.gripper_ratio.append(open_ratio)
        episode.gripper_position.append(raw_gripper)
        episode.gripper_closed.append(float(gripper_closed))
        episode.robot_timestamps.append(float(robot_time_s))
        episode.pc_timestamps.append(pc_time_s)
        episode.sample_indices.append(sample_index)
        episode.last_accept_time_s = now

        self._show_live_preview(image_main, image_wrist, sample_index, raw_gripper)
        return episode.sample_count

    def end_episode(self, success_flag: int = 1) -> int:
        """结束当前轨迹，并触发保存/删除确认。"""
        if self.current_episode is None:
            print("[rpc] end_episode 被忽略：当前没有活动轨迹")
            return 0

        episode = self.current_episode
        print(
            "[rpc] end_episode -> "
            f"episode_{episode.episode_index:04d} samples={episode.sample_count} success={success_flag}"
        )

        if episode.sample_count == 0:
            print("[rpc] 当前轨迹没有收到任何样本，直接丢弃")
            self.current_episode = None
            self.episode_index += 1
            return 0

        save_path, preview_path = self._write_episode(episode)
        print(f"[save] npz -> {save_path}")
        print(f"[save] preview -> {preview_path}")

        keep_episode = self._prompt_keep_episode(save_path, episode.sample_count)
        cv2.destroyWindow("Episode Review")

        if keep_episode:
            print(f"[save] 已保留 {save_path.name}")
            result = 1
        else:
            save_path.unlink(missing_ok=True)
            preview_path.unlink(missing_ok=True)
            print(f"[save] 已删除 {save_path.name}")
            result = 0

        self.current_episode = None
        self.episode_index += 1
        return result


def render_ur_script(runtime_config: dict, local_ip: str, rpc_port: int) -> str:
    """根据配置自动生成最小化的示教器端 URScript 模板。

    示教器端只负责上传三类量：
    - 夹爪角度 / 开合量
    - TCP 六维绝对位姿
    - 六个关节的相对角度
    """
    ur_cfg = runtime_config["ur_script"]

    rpc_url = ur_cfg["rpc_url"]
    if rpc_url == "auto":
        rpc_url = f"http://{local_ip}:{rpc_port}/RPC2"

    prefix = ur_cfg["gripper_prefix"]
    gripper_index = int(ur_cfg["gripper_index"])
    activate_gripper = bool(ur_cfg["activate_gripper"])
    activate_lines = []
    if activate_gripper:
        activate_lines.append(f"  {prefix}_set_activate({gripper_index})")
    activate_lines.extend(
        [
            f"  {prefix}_set_force({gripper_index}, {float(ur_cfg['gripper_force']):.1f})",
            f"  {prefix}_set_speed({gripper_index}, {float(ur_cfg['gripper_speed']):.1f})",
            "  sleep(0.5)",
        ]
    )

    activate_block = "\n".join(activate_lines)

    script = f"""# 该脚本由 auto_collect_rpc.py 自动生成。
# Script 节点纯顺序版本，不使用 def。
# 示教器端只上传三类数据：
# 1. current_gripper_pos: 夹爪角度 / 开合量
# 2. get_actual_tcp_pose(): 六维绝对位姿 [x, y, z, rx, ry, rz]
# 3. get_actual_joint_positions(): 六个关节相对角度
global rpc_url = "{rpc_url}"
global collection_active = True
global current_gripper_pos = {float(ur_cfg["gripper_open_position"]):.1f}
global rpc_handle = rpc_factory("xmlrpc", rpc_url)
{activate_block}
rpc_handle.begin_episode()

# 如果你在别处控制夹爪，请同步更新 current_gripper_pos
# 例如：
# current_gripper_pos = 20.0
# {prefix}_set_position({gripper_index}, current_gripper_pos)

while collection_active:
  rpc_handle.push_sample(
    get_actual_tcp_pose(),
    get_actual_joint_positions(),
    current_gripper_pos
  )
  sync()
end

rpc_handle.end_episode(1)
rpc_handle.closeXMLRPCClientConnection()
"""
    return script


def write_generated_ur_files(runtime_config: dict, local_ip: str, rpc_port: int) -> tuple[Path, Path]:
    """把根据配置生成的示教器脚本和说明写入磁盘。"""
    example_dir = Path(__file__).resolve().parent.parent / "example_code_in_ur"
    script_path = example_dir / "auto_collect_rpc_program.script"
    md_path = example_dir / "auto_collect_rpc_program.md"

    script_content = render_ur_script(runtime_config, local_ip, rpc_port)
    script_path.write_text(script_content, encoding="utf-8")

    md_content = f"""# 示教器端脚本说明

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
http://{local_ip}:{rpc_port}/RPC2
```

当前配置文件：

```text
{DEFAULT_CONFIG_PATH}
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
"""
    md_path.write_text(md_content, encoding="utf-8")
    return script_path, md_path


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="配置文件路径，默认使用 data_fetch_v2/configs/auto_collect_config.json",
    )
    parser.add_argument(
        "--no-camera-preview",
        action="store_true",
        default=False,
        help="临时关闭相机顺序预览，仅影响本次运行",
    )
    return parser


def main() -> None:
    """程序入口。"""
    args = build_parser().parse_args()
    runtime_config, config_path = load_config(args.config)

    if args.no_camera_preview:
        runtime_config["cameras"]["preview_enabled"] = False

    robot_ip = runtime_config["robot"]["ip"]
    rpc_host = runtime_config["rpc"]["host"]
    rpc_port = int(runtime_config["rpc"]["port"])

    service = AutoCollectRPCService(runtime_config, config_path)
    server = CompatibleXMLRPCServer(
        (rpc_host, rpc_port),
        requestHandler=RPCRequestHandler,
        allow_none=True,
        logRequests=False,
    )
    server.register_function(service.ping, "ping")
    server.register_function(service.begin_episode, "begin_episode")
    server.register_function(service.push_sample, "push_sample")
    server.register_function(service.end_episode, "end_episode")

    local_ip = guess_local_ip(robot_ip)
    script_path, md_path = write_generated_ur_files(runtime_config, local_ip, rpc_port)

    print("=" * 72)
    print("XVLA 自动采集服务已就绪")
    print(f"配置文件 : {config_path}")
    print(f"XML-RPC  : http://{local_ip}:{rpc_port}/RPC2")
    print(f"保存目录 : {service.save_dir}")
    print(f"机器人IP : {robot_ip}")
    print(f"示教器脚本 : {script_path}")
    print(f"脚本说明 : {md_path}")
    print("RPC 流程 : begin_episode -> push_sample -> end_episode")
    print("=" * 72)

    dashboard_cfg = runtime_config["dashboard"]
    if bool(dashboard_cfg["auto_start_program"]):
        print(f"[dashboard] 尝试加载并启动 {dashboard_cfg['program_name']}")
        DashboardClient(robot_ip).load_and_play(str(dashboard_cfg["program_name"]))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 用户已停止服务")
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
