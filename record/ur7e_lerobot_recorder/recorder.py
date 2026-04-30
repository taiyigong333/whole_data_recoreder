from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import CameraSpec
from .dataset import RawSample, create_or_resume_dataset, dataset_frame_from_samples
from .runtime import (
    RTDEReceiveInterface,
    cv2,
    get_camera_config_types,
    get_camera_factory,
    get_video_encoding_manager_cls,
    pyrealsense2_available,
)


class GripperReader:
    """从 UR 的数字量寄存器或人工标注中读取夹爪状态。"""

    def __init__(self, rtde_r, source: str, index: int, initial_manual_value: float = 0.0):
        self.rtde_r = rtde_r
        self.source = source
        self.index = index
        self.manual_value = float(initial_manual_value)
        self.last_valid_value: float | None = (
            self.manual_value if source == "manual" else None
        )
        self._warned_fallback = False

    def set_manual(self, value: float) -> None:
        self.manual_value = float(value)
        self.last_valid_value = self.manual_value

    def switch_to_manual(self, value: float | None = None) -> None:
        self.source = "manual"
        if value is not None:
            self.manual_value = float(value)
        self.last_valid_value = self.manual_value

    def read(self) -> float:
        if self.source == "manual":
            return self.manual_value

        try:
            if self.source == "do":
                bits = self.rtde_r.getActualDigitalOutBits()
            elif self.source == "di":
                bits = self.rtde_r.getActualDigitalInputBits()
            elif self.source == "tool_do":
                bits = self.rtde_r.getToolDigitalOutBits()
            elif self.source == "tool_di":
                bits = self.rtde_r.getToolDigitalInputBits()
            else:
                raise ValueError(f"不支持的夹爪读取来源: {self.source}")

            value = float((bits >> self.index) & 1)
            self.last_valid_value = value
            return value
        except Exception as exc:  # pragma: no cover - hardware dependency
            if self.last_valid_value is None:
                raise RuntimeError(
                    "读取夹爪状态失败，且当前没有可用的回退值。"
                ) from exc

            if not self._warned_fallback:
                print(
                    "[警告] 本帧夹爪状态读取失败一次，将复用上一帧读取到的有效值。"
                )
                self._warned_fallback = True
            return self.last_valid_value


def _rotation_enum(rotation: int, cv2_rotation_enum):
    mapping = {
        0: cv2_rotation_enum.NO_ROTATION,
        90: cv2_rotation_enum.ROTATE_90,
        180: cv2_rotation_enum.ROTATE_180,
        -90: cv2_rotation_enum.ROTATE_270,
    }
    return mapping[rotation]


def build_camera_configs(camera_specs: Sequence[CameraSpec]) -> dict[str, Any]:
    OpenCVCameraConfig, RealSenseCameraConfig, ColorMode, Cv2Rotation = get_camera_config_types()
    configs: dict[str, Any] = {}

    for spec in camera_specs:
        rotation = _rotation_enum(spec.rotation, Cv2Rotation)
        if spec.backend == "opencv":
            device = spec.device
            if isinstance(device, str) and not device.isdigit():
                device = Path(device).expanduser()
            configs[spec.name] = OpenCVCameraConfig(
                index_or_path=device,
                fps=spec.stream_fps,
                width=spec.width,
                height=spec.height,
                color_mode=ColorMode.RGB,
                rotation=rotation,
                warmup_s=spec.warmup_s,
            )
        else:
            configs[spec.name] = RealSenseCameraConfig(
                serial_number_or_name=str(spec.device),
                fps=spec.stream_fps,
                width=spec.width,
                height=spec.height,
                color_mode=ColorMode.RGB,
                use_depth=spec.use_depth,
                rotation=rotation,
                warmup_s=spec.warmup_s,
            )

    return configs


def scan_available_cameras(scan_camera_type: str, max_camera_index: int) -> None:
    if scan_camera_type in {"all", "opencv"}:
        from lerobot.cameras.opencv import OpenCVCamera

        print("正在扫描 OpenCV 相机...")
        cameras = []
        for info in OpenCVCamera.find_cameras():
            camera_id = info.get("id")
            if isinstance(camera_id, int) and camera_id > max_camera_index:
                continue
            cameras.append(info)

        if not cameras:
            print("  没有扫描到可用的 OpenCV 相机。")
        else:
            for info in cameras:
                profile = info.get("default_stream_profile", {})
                print(
                    "  "
                    f"id={info.get('id')} "
                    f"backend={info.get('backend_api')} "
                    f"default={profile.get('width')}x{profile.get('height')}@{profile.get('fps')}"
                )

    if scan_camera_type in {"all", "intelrealsense"}:
        if not pyrealsense2_available():
            print(
                "跳过 RealSense 扫描：当前环境缺少 pyrealsense2。"
            )
            return

        from lerobot.cameras.realsense import RealSenseCamera

        print("正在扫描 RealSense 相机...")
        cameras = RealSenseCamera.find_cameras()
        if not cameras:
            print("  没有扫描到可用的 RealSense 相机。")
            return

        for info in cameras:
            profile = info.get("default_stream_profile", {})
            print(
                "  "
                f"name={info.get('name')} "
                f"serial={info.get('id')} "
                f"default={profile.get('width')}x{profile.get('height')}@{profile.get('fps')}"
            )


def _maybe_set_opencv_buffer(camera: Any, buffer_size: int) -> None:
    videocapture = getattr(camera, "videocapture", None)
    if videocapture is None:
        return
    try:
        videocapture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    except Exception:  # pragma: no cover - best effort
        pass


def open_cameras(camera_specs: Sequence[CameraSpec], camera_buffer_size: int) -> dict[str, Any]:
    make_cameras_from_configs = get_camera_factory()
    cameras = make_cameras_from_configs(build_camera_configs(camera_specs))

    try:
        for spec in camera_specs:
            camera = cameras[spec.name]
            camera.connect()
            if spec.backend == "opencv":
                _maybe_set_opencv_buffer(camera, camera_buffer_size)
            frame = camera.read()
            print(
                f"已打开相机 '{spec.name}' ({spec.backend}:{spec.device})，"
                f"首帧分辨率为 {frame.shape[1]}x{frame.shape[0]}。"
            )
    except BaseException:
        close_cameras(cameras)
        raise

    return cameras


def close_cameras(cameras: dict[str, Any]) -> None:
    for camera in cameras.values():
        try:
            if getattr(camera, "is_connected", False):
                camera.disconnect()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def capture_synchronized(
    rtde_r,
    cameras: dict[str, Any],
    camera_specs: Sequence[CameraSpec],
    gripper_reader: GripperReader,
    sync_mode: str,
) -> RawSample | None:
    robot_state: dict[str, np.ndarray | float] = {}
    images: dict[str, np.ndarray] = {}
    capture_times: list[float] = []
    errors: list[str] = []
    expected_names = [spec.name for spec in camera_specs]

    def robot_worker() -> None:
        try:
            robot_state["joints"] = np.asarray(rtde_r.getActualQ(), dtype=np.float32)
            robot_state["tcp_pose"] = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float32)
            robot_state["gripper"] = float(gripper_reader.read())
            capture_times.append(time.perf_counter())
        except Exception as exc:  # pragma: no cover - hardware dependency
            errors.append(f"机器人状态读取失败: {exc}")

    def camera_worker(name: str) -> None:
        try:
            camera = cameras[name]
            if sync_mode == "async":
                frame = camera.async_read(timeout_ms=500)
            else:
                frame = camera.read()
            images[name] = frame
            capture_times.append(time.perf_counter())
        except Exception as exc:  # pragma: no cover - hardware dependency
            errors.append(f"相机 '{name}' 读取失败: {exc}")

    if sync_mode == "threaded":
        threads = [threading.Thread(target=robot_worker, daemon=True)]
        for name in expected_names:
            threads.append(threading.Thread(target=camera_worker, args=(name,), daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        robot_worker()
        for name in expected_names:
            camera_worker(name)

    if errors:
        print("[警告] " + " | ".join(errors))
        return None

    if set(images) != set(expected_names):
        print("[警告] 本帧有相机图像缺失，已跳过。")
        return None

    joints = robot_state.get("joints")
    tcp_pose = robot_state.get("tcp_pose")
    gripper = robot_state.get("gripper")
    if joints is None or tcp_pose is None or gripper is None:
        print("[警告] 本帧机器人状态不完整，已跳过。")
        return None

    skew_ms = 0.0
    if capture_times:
        skew_ms = (max(capture_times) - min(capture_times)) * 1000.0

    return RawSample(
        joints=joints,
        tcp_pose=tcp_pose,
        gripper=float(gripper),
        images=images,
        skew_ms=skew_ms,
    )


def overlay_text(image_bgr: np.ndarray, text: str, row: int) -> None:
    y = 24 + row * 22
    cv2.putText(
        image_bgr,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def _preview_tile(image_rgb: np.ndarray, label: str, width: int = 320, height: int = 240) -> np.ndarray:
    tile = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(
        tile,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return tile


def _compose_preview_grid(tiles: list[np.ndarray]) -> np.ndarray:
    if len(tiles) == 1:
        return tiles[0]

    rows = []
    for index in range(0, len(tiles), 2):
        row_tiles = tiles[index : index + 2]
        if len(row_tiles) == 1:
            blank = np.zeros_like(row_tiles[0])
            row_tiles.append(blank)
        rows.append(cv2.hconcat(row_tiles))
    return cv2.vconcat(rows)


def build_preview(
    sample: RawSample,
    camera_specs: Sequence[CameraSpec],
    episode_index: int,
    transition_count: int,
    raw_count: int,
    fps: int,
    gripper_source: str,
) -> np.ndarray:
    tiles = [_preview_tile(sample.images[spec.name], spec.name) for spec in camera_specs]
    mosaic = _compose_preview_grid(tiles)
    footer = np.zeros((96, mosaic.shape[1], 3), dtype=np.uint8)

    grip_text = "闭合" if sample.gripper > 0.5 else "打开"
    overlay_text(
        footer,
        f"轨迹={episode_index} 原始帧={raw_count} 转移={transition_count} fps={fps}",
        row=0,
    )
    overlay_text(
        footer,
        (
            f"tcp_xyz=[{sample.tcp_pose[0]:+.3f}, {sample.tcp_pose[1]:+.3f}, "
            f"{sample.tcp_pose[2]:+.3f}] 夹爪={grip_text} 来源={gripper_source}"
        ),
        row=1,
    )
    overlay_text(
        footer,
        f"采集偏差毫秒={sample.skew_ms:.1f} 按键: q/esc 结束 c 闭合 o 或 g 打开",
        row=2,
    )
    return cv2.vconcat([mosaic, footer])


def read_manual_key(key: int, gripper_reader: GripperReader, sample: RawSample) -> None:
    if key == ord("c"):
        gripper_reader.set_manual(1.0)
        sample.gripper = 1.0
    elif key in (ord("o"), ord("g")):
        gripper_reader.set_manual(0.0)
        sample.gripper = 0.0


def wait_for_episode_start(next_episode_index: int) -> bool:
    text = (
        f"\n第 {next_episode_index} 条轨迹：请先把机器人移到起始位并开启 Freedrive，"
        "准备好后按回车开始；输入 q 后回车可以退出："
    )
    answer = input(text).strip().lower()
    return answer not in {"q", "quit", "exit"}


def finalize_episode(dataset, transition_count: int, min_transitions: int) -> str:
    if transition_count < min_transitions:
        print(
            f"这条轨迹太短（{transition_count} 个转移样本，小于 {min_transitions}），已丢弃。"
        )
        dataset._wait_image_writer()
        dataset.clear_episode_buffer()
        return "discard"

    while True:
        answer = input(
            "请选择：[s] 保存并继续 / [q] 保存并退出 / [d] 丢弃并继续 / [x] 丢弃并退出："
        ).strip().lower()
        if answer in {"s", "q", "d", "x"}:
            break

    if answer in {"d", "x"}:
        dataset._wait_image_writer()
        dataset.clear_episode_buffer()
        print("当前轨迹已丢弃。")
        return "discard_and_quit" if answer == "x" else "discard"

    print("正在编码并保存当前轨迹...")
    dataset.save_episode()
    print(
        f"已保存轨迹 {dataset.num_episodes - 1}，共 {transition_count} 个转移样本，目录为 {dataset.root}"
    )
    return "save_and_quit" if answer == "q" else "save"


def record_episode(
    dataset,
    rtde_r,
    cameras: dict[str, Any],
    camera_specs: Sequence[CameraSpec],
    gripper_reader: GripperReader,
    args,
) -> tuple[str, int]:
    previous_sample: RawSample | None = None
    raw_count = 0
    transition_count = 0
    interval_s = 1.0 / args.fps

    while raw_count < args.max_frames:
        loop_start = time.perf_counter()
        current_sample = capture_synchronized(
            rtde_r=rtde_r,
            cameras=cameras,
            camera_specs=camera_specs,
            gripper_reader=gripper_reader,
            sync_mode=args.sync_mode,
        )
        if current_sample is None:
            time.sleep(min(0.05, interval_s))
            continue

        raw_count += 1

        key = -1
        if args.preview:
            preview = build_preview(
                sample=current_sample,
                camera_specs=camera_specs,
                episode_index=dataset.num_episodes,
                transition_count=transition_count,
                raw_count=raw_count,
                fps=args.fps,
                gripper_source=gripper_reader.source,
            )
            cv2.imshow("UR7e LeRobotDataset 采集器", preview)
            key = cv2.waitKey(1) & 0xFF
            if gripper_reader.source == "manual":
                read_manual_key(key, gripper_reader, current_sample)

        if previous_sample is not None:
            frame = dataset_frame_from_samples(previous_sample, current_sample, camera_specs)
            dataset.add_frame(frame=frame, task=args.task)
            transition_count += 1

        previous_sample = current_sample

        if key in {ord("q"), 27}:
            break

        dt_s = time.perf_counter() - loop_start
        if dt_s < interval_s:
            time.sleep(interval_s - dt_s)

    outcome = finalize_episode(dataset, transition_count, args.min_transitions)
    return outcome, transition_count


def maybe_switch_gripper_to_manual(args, gripper_reader: GripperReader) -> None:
    if gripper_reader.source == "manual":
        print("当前夹爪模式为人工标注。按 c 表示闭合，按 o 或 g 表示打开。")
        return

    try:
        initial = gripper_reader.read()
        status = "闭合" if initial > 0.5 else "打开"
        print(
            f"已成功读取初始夹爪状态：{status} "
            f"(来源={gripper_reader.source}[{gripper_reader.index}])。"
        )
    except Exception as exc:  # pragma: no cover - hardware dependency
        if not args.preview:
            raise SystemExit(
                "自动读取夹爪失败，且当前关闭了预览窗口，无法切换到人工标注模式。"
            ) from exc
        print(
            "[警告] 启动阶段自动读取夹爪失败，将切换到人工标注模式。"
            "按 c 表示闭合，按 o 或 g 表示打开。"
        )
        gripper_reader.switch_to_manual(args.manual_gripper_initial)


def _print_camera_summary(camera_specs: Sequence[CameraSpec]) -> None:
    print("本次采集使用的相机配置：")
    for spec in camera_specs:
        stream_fps = spec.stream_fps if spec.stream_fps is not None else "default"
        print(
            f"  - {spec.name}: {spec.backend}:{spec.device} "
            f"{spec.width}x{spec.height} stream_fps={stream_fps}"
        )


def run_recording(args, camera_specs: Sequence[CameraSpec]) -> None:
    dataset = None
    rtde_r = None
    cameras: dict[str, Any] = {}
    saved_episodes = 0

    try:
        dataset = create_or_resume_dataset(args, camera_specs)
        print(f"正在连接 UR7e，目标地址为 {args.robot_ip} ...")
        rtde_r = RTDEReceiveInterface(args.robot_ip)
        print(f"连接成功。当前 TCP 位姿: {rtde_r.getActualTCPPose()}")

        _print_camera_summary(camera_specs)
        cameras = open_cameras(camera_specs, args.camera_buffer_size)
        gripper_reader = GripperReader(
            rtde_r=rtde_r,
            source=args.gripper_source,
            index=args.gripper_index,
            initial_manual_value=args.manual_gripper_initial,
        )
        maybe_switch_gripper_to_manual(args, gripper_reader)

        print("\n录制规则：")
        print("  1. UR7e 必须保持在本地手动模式，并开启 Freedrive。")
        print("  2. 脚本只读取状态，不会向机器人发送运动命令。")
        print("  3. 在预览窗口里按 q 或 Esc，可结束当前轨迹。")
        if gripper_reader.source == "manual":
            print("  4. 人工夹爪标注时：按 c 表示闭合，按 o 或 g 表示打开。")

        VideoEncodingManager = get_video_encoding_manager_cls()
        with VideoEncodingManager(dataset):
            while args.num_episodes == 0 or saved_episodes < args.num_episodes:
                if not wait_for_episode_start(dataset.num_episodes):
                    break

                outcome, transition_count = record_episode(
                    dataset=dataset,
                    rtde_r=rtde_r,
                    cameras=cameras,
                    camera_specs=camera_specs,
                    gripper_reader=gripper_reader,
                    args=args,
                )

                if outcome.startswith("save"):
                    saved_episodes += 1
                    print(
                        f"当前数据集概况：轨迹数={dataset.num_episodes}，样本帧数={dataset.num_frames}"
                    )
                elif transition_count == 0:
                    print("这条轨迹里没有采到有效的转移样本。")

                if outcome in {"save_and_quit", "discard_and_quit"}:
                    break

        if args.push_to_hub and dataset.num_episodes > 0:
            print("正在将数据集推送到 Hugging Face Hub ...")
            dataset.push_to_hub(private=args.private)
            print("数据集推送完成。")

        print(
            f"采集结束。数据集目录: {dataset.root} | 轨迹数={dataset.num_episodes} | 样本帧数={dataset.num_frames}"
        )
    finally:
        if cameras:
            close_cameras(cameras)
        if cv2 is not None:  # pragma: no branch
            cv2.destroyAllWindows()
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.stop_image_writer()
