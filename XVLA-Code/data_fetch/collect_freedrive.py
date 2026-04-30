"""
UR7e 自由拖动示教数据采集（双 RealSense 摄像头）
仅需 RTDEReceiveInterface（只读），不需要远程控制权限
使用前：示教器上切换到本地手动模式，开启自由驱动（Freedrive）
摄像头: Intel RealSense D435i (主视角) + D405 (腕部视角)
"""

import cv2
import time
import threading
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from rtde_receive import RTDEReceiveInterface

# ============ 配置（根据你的环境修改） ============
ROBOT_IP = "192.168.1.88"       # UR7e 控制柜 IP
SAVE_DIR = Path("./raw_demos") # 原始数据保存目录
FPS = 30                        # 采集帧率
TASK_NAME = "Pick up the chili on the table"  # 语言指令
MAX_FRAMES = 900                # 单条轨迹最大帧数（30fps × 30秒）
# ==============================

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def start_realsense_pipelines():
    """扫描 RealSense 设备并为每个设备启动彩色流 pipeline"""
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) < 2:
        raise RuntimeError(f"需要 2 个 RealSense 摄像头，只找到 {len(devices)} 个")

    print(f"找到 {len(devices)} 个 RealSense 设备:")
    pipelines = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  - {name} (S/N: {serial})")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)

        try:
            pipeline.start(config)
            pipelines.append({
                'name': name,
                'serial': serial,
                'pipe': pipeline,
            })
            print(f"  ▶ {name} 启动成功")
        except Exception as e:
            print(f"  ✗ {name} 启动失败: {e}")
            # 关闭已启动的
            for p in pipelines:
                p['pipe'].stop()
            raise RuntimeError(f"{name} 启动失败: {e}")

    return pipelines


def capture_aligned(rtde_r, pipelines):
    """并行采集两个视角图像和位姿"""
    pose_holder = [None]

    def read_pose():
        pose_holder[0] = rtde_r.getActualTCPPose()

    t = threading.Thread(target=read_pose)
    t.start()

    frames = []
    for p in pipelines:
        try:
            fs = p['pipe'].wait_for_frames(timeout_ms=2000)
            cf = fs.get_color_frame()
            if cf:
                frames.append(np.asanyarray(cf.get_data()))
            else:
                frames.append(None)
        except RuntimeError:
            frames.append(None)

    t.join()

    if pose_holder[0] is None or any(f is None for f in frames):
        return None, None, None
    return frames[0], frames[1], pose_holder[0]


def main():
    # ---- 初始化 ----
    print("正在连接机器人（只读模式）...")
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    print(f"机器人已连接，当前位姿: {rtde_r.getActualTCPPose()}")

    print("\n正在启动 RealSense 摄像头...")
    pipelines = start_realsense_pipelines()
    pipe_main = pipelines[0]
    pipe_wrist = pipelines[1]
    print(f"主视角: {pipe_main['name']} (S/N: {pipe_main['serial']})")
    print(f"腕部视角: {pipe_wrist['name']} (S/N: {pipe_wrist['serial']})")

    # ---- 交互式确认哪个是哪个 ----
    print("\n正在预览摄像头画面，请确认视角分配...")
    print("按 's' 交换两个视角，按 Enter 确认继续")
    while True:
        try:
            f1 = pipe_main['pipe'].wait_for_frames(timeout_ms=2000).get_color_frame()
            f2 = pipe_wrist['pipe'].wait_for_frames(timeout_ms=2000).get_color_frame()
            if f1 and f2:
                img1 = np.asanyarray(f1.get_data())
                img2 = np.asanyarray(f2.get_data())
                cv2.putText(img1, f"Main: {pipe_main['name']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img2, f"Wrist: {pipe_wrist['name']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                preview = np.hstack([img1, img2])
                cv2.imshow("Camera Preview - Enter=confirm, S=swap", preview)
        except RuntimeError:
            pass
        key = cv2.waitKey(100) & 0xFF
        if key == 13:  # Enter
            break
        elif key == ord('s'):
            pipe_main, pipe_wrist = pipe_wrist, pipe_main
            print(f"已交换 -> 主视角: {pipe_main['name']}, 腕部视角: {pipe_wrist['name']}")
    cv2.destroyAllWindows()

    active_pipelines = [pipe_main, pipe_wrist]

    # 夹爪状态（键盘标记）
    gripper_state = 0.0  # 0.0=闭合, 1.0=打开

    print("\n" + "=" * 50)
    print("UR7e 自由拖动采集（双 RealSense 摄像头）")
    print("=" * 50)
    print("操作步骤：")
    print("  0. 示教器上进入 本地手动模式 → 开启自由驱动(Freedrive)")
    print("  1. 按住示教器背面按钮，手拖机器人")
    print("  2. 按 'g' 标记夹爪打开，按 'c' 标记闭合")
    print("  3. 按 'q' 或 ESC 结束当前轨迹")
    print("  4. 按 Enter 开始下一条")
    print("=" * 50)

    episode = 0
    interval = 1.0 / FPS

    try:
        while True:
            print(f"\n--- 轨迹 #{episode} ---")
            print("按住示教器自由驱动按钮，将机器人移到起始位置")
            input("准备好后按 Enter 开始录制...")
            print(">>> 录制已开始！按 q 结束 <<<")

            images_main = []
            images_wrist = []
            tcp_poses = []
            gripper_states = []
            frame_count = 0

            while frame_count < MAX_FRAMES:
                loop_start = time.time()

                # ① 并行采集：两个视角 + 位姿
                frame_m, frame_w, tcp = capture_aligned(rtde_r, active_pipelines)
                if frame_m is None:
                    continue

                frame_m_rgb = cv2.cvtColor(frame_m, cv2.COLOR_BGR2RGB)
                frame_w_rgb = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)
                images_main.append(cv2.resize(frame_m_rgb, (256, 256)))
                images_wrist.append(cv2.resize(frame_w_rgb, (256, 256)))
                tcp_poses.append(tcp)

                # ② 记录夹爪状态
                gripper_states.append(gripper_state)
                frame_count += 1

                # ③ 显示预览（左右并排）
                display_m = frame_m.copy()
                display_w = frame_w.copy()

                cv2.putText(display_m, f"Ep#{episode} F:{frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_m,
                            f"TCP: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                grip_str = "OPEN" if gripper_state > 0.5 else "CLOSED"
                grip_color = (0, 255, 0) if gripper_state > 0.5 else (0, 0, 255)
                cv2.putText(display_m, f"Gripper: {grip_str}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 2)

                # 底部操作提示（高亮显示当前可执行的操作）
                if gripper_state > 0.5:
                    hint = "REC | G=opened | Press C to CLOSE | Q=stop"
                else:
                    hint = "REC | C=closed | Press G to OPEN | Q=stop"
                cv2.putText(display_m, hint, (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display_w, "Wrist", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                display = np.hstack([display_m, display_w])
                cv2.imshow("UR7e Collection", display)

                key = cv2.waitKey(1) & 0xFF

                # 夹爪标记
                if key == ord('c'):
                    gripper_state = 0.0
                    gripper_states[-1] = 0.0
                    print("  标记: 夹爪 → 闭合")
                elif key == ord('g'):
                    gripper_state = 1.0
                    gripper_states[-1] = 1.0
                    print("  标记: 夹爪 → 打开")

                if key == ord('q') or key == 27:
                    break

                # 控制采集帧率
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

            # ---- 保存轨迹 ----
            if len(images_main) < 30:
                print(f"轨迹太短（{len(images_main)} 帧），丢弃")
                continue

            np.savez_compressed(
                SAVE_DIR / f"episode_{episode:04d}.npz",
                images=np.array(images_main, dtype=np.uint8),
                images_wrist=np.array(images_wrist, dtype=np.uint8),
                tcp_poses=np.array(tcp_poses, dtype=np.float64),
                gripper=np.array(gripper_states, dtype=np.float64),
                instruction=TASK_NAME,
                fps=FPS,
            )
            print(f"已保存: episode_{episode:04d}.npz ({len(images_main)} 帧)")

            episode += 1

            cont = input("继续采集下一条？(y/n): ")
            if cont.lower() != 'y':
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        for p in active_pipelines:
            p['pipe'].stop()
        cv2.destroyAllWindows()
        print(f"采集结束，共保存 {episode} 条轨迹到 {SAVE_DIR}")


if __name__ == "__main__":
    main()
