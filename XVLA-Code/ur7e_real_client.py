"""
UR7e 真机推理客户端 — 连接云端 XVLA 推理服务
摄像头: RealSense D435i (主视角) + D405 (腕部视角)
机器人: UR7e (RTDE 控制接口)
夹爪: XML-RPC 控制示教器 UR 程序 (0=闭合, 100=张开)
"""

import argparse, time, signal, os, socket, threading
import numpy as np
import cv2
import json_numpy
import requests
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from xmlrpc.server import SimpleXMLRPCServer
from rtde_receive import RTDEReceiveInterface
from collections import deque


ROBOT_IP = "192.168.1.88"
DASHBOARD_PORT = 29999
URSCRIPT_PORT = 30002
CAMERA_FPS = 30
CONTROL_HZ = 5
CONTROL_DT = 1.0 / CONTROL_HZ
MAX_POS_DELTA = 0.1
MAX_ROT_DELTA = 0.5


# ============ 夹爪 XML-RPC 控制 ============

gripper_target = 100.0

def get_gripper_target():
    return float(gripper_target)

def run_gripper_server():
    server = SimpleXMLRPCServer(("0.0.0.0", 50000), allow_none=True, logRequests=False)
    server.register_function(get_gripper_target, "get_pos")
    server.serve_forever()

def send_dashboard_command(command):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect((ROBOT_IP, DASHBOARD_PORT))
    s.recv(1024)
    if not command.endswith("\n"):
        command += "\n"
    s.sendall(command.encode("utf-8"))
    s.recv(1024)
    s.close()

def start_gripper():
    global gripper_target
    gripper_target = 100.0
    t = threading.Thread(target=run_gripper_server, daemon=True)
    t.start()
    time.sleep(0.5)
    send_dashboard_command("load test_xmlrpc.urp")
    time.sleep(1)
    send_dashboard_command("play")
    print("  夹爪服务已启动 (XML-RPC 端口 50000)")

def set_gripper(open_flag):
    global gripper_target
    gripper_target = 100.0 if open_flag else 0.0


# ============ RTDE 连接管理 ============

class RTDEConnection:
    def __init__(self, ip):
        self.ip = ip
        self.recv = None
        self.connect()

    def connect(self):
        print(f"  连接 RTDE (只读): {self.ip} ...")
        self.recv = RTDEReceiveInterface(self.ip)
        tcp = self.recv.getActualTCPPose()
        print(f"  RTDE 已连接, TCP: {[f'{v:.4f}' for v in tcp]}")
        return True

    def is_connected(self):
        try:
            self.recv.getActualTCPPose()
            return True
        except Exception:
            return False

    def reconnect(self):
        print("  RTDE 连接断开，尝试重连...")
        try:
            self.recv = RTDEReceiveInterface(self.ip)
            tcp = self.recv.getActualTCPPose()
            print(f"  RTDE 重连成功, TCP: {[f'{v:.4f}' for v in tcp]}")
            return True
        except Exception as e:
            print(f"  RTDE 重连失败: {e}")
            return False

    def get_tcp(self):
        return self.recv.getActualTCPPose()

    def moveL(self, target, speed=0.3, accel=0.3, grip_pos=None):
        """通过 URScript 发送 moveL + 夹爪控制"""
        x, y, z, rx, ry, rz = target
        cmd = f"movel(p[{x},{y},{z},{rx},{ry},{rz}],{speed},{accel})"
        if grip_pos is not None:
            cmd += f"\ndh_ag95_set_position(1, {grip_pos})"
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((self.ip, URSCRIPT_PORT))
        s.sendall((cmd + "\n").encode("utf-8"))
        s.close()


# ============ 工具函数 ============

def rotate6d_to_rotmat(v6):
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def tcp_to_proprio(tcp_pose, grip_state=1.0):
    pos = np.array(tcp_pose[:3])
    rotmat = R.from_rotvec(tcp_pose[3:6]).as_matrix()
    rot6d = rotmat[:, :2].flatten()
    left = np.concatenate([pos, rot6d, [grip_state]])
    right = np.zeros(10)
    return np.concatenate([left, right]).astype(np.float32)


def parse_action(action_20d):
    left = action_20d[:10]
    pos = left[:3]
    rot6d = left[3:9]
    grip_logit = left[9]
    rotmat = rotate6d_to_rotmat(rot6d)
    rotvec = R.from_matrix(rotmat).as_rotvec()
    return pos, rotvec, grip_logit


def check_action_safety(current_tcp, target_tcp):
    delta_pos = np.linalg.norm(np.array(target_tcp[:3]) - np.array(current_tcp[:3]))
    delta_rot = np.linalg.norm(np.array(target_tcp[3:6]) - np.array(current_tcp[3:6]))
    if delta_pos > MAX_POS_DELTA:
        print(f"  ⚠ 跳过: 位移过大 {delta_pos:.4f}m > {MAX_POS_DELTA}m")
        return False
    if delta_rot > MAX_ROT_DELTA:
        print(f"  ⚠ 跳过: 旋转过大 {delta_rot:.4f}rad > {MAX_ROT_DELTA}rad")
        return False
    return True


# ============ 摄像头 ============

def start_realsense():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        raise RuntimeError(f"需要 2 个 RealSense，只找到 {len(devices)} 个")

    pipelines = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipeline.start(config)
        pipelines.append({"name": name, "serial": serial, "pipe": pipeline})
        print(f"  摄像头: {name} (S/N: {serial})")

    return pipelines


def capture_images(pipelines):
    frames = []
    for p in pipelines:
        try:
            fs = p["pipe"].wait_for_frames(timeout_ms=2000)
            cf = fs.get_color_frame()
            if cf:
                img = np.asanyarray(cf.get_data())
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(img_rgb, (256, 256)))
            else:
                frames.append(None)
        except RuntimeError:
            frames.append(None)

    if any(f is None for f in frames):
        return None, None
    return frames[0], frames[1]


# ============ 推理客户端 ============

class XVLAClient:
    def __init__(self, host="localhost", port=8010, domain_id=12, steps=10, use_https=False):
        scheme = "https" if use_https else "http"
        self.url = f"{scheme}://{host}:{port}/act"
        self.domain_id = domain_id
        self.steps = steps

    def predict(self, image_main, image_wrist, proprio, instruction):
        payload = {
            "image0": json_numpy.dumps(image_main),
            "image1": json_numpy.dumps(image_wrist),
            "proprio": json_numpy.dumps(proprio),
            "language_instruction": instruction,
            "domain_id": self.domain_id,
            "steps": self.steps,
        }
        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Server error: {data['error']}")
        return np.array(data["action"], dtype=np.float32)


# ============ 主循环 ============

def run(args):
    print("=" * 60)
    print("UR7e 真机推理客户端")
    print(f"控制频率: {CONTROL_HZ}Hz (间隔 {CONTROL_DT*1000:.0f}ms)")
    print(f"安全阈值: 位移 {MAX_POS_DELTA*100:.0f}cm, 旋转 {MAX_ROT_DELTA:.2f}rad")
    print("=" * 60)

    # 连接机器人
    print("\n[1/3] 连接 UR7e...")
    rtde = RTDEConnection(ROBOT_IP)

    # 初始化夹爪
    print("\n[2/3] 初始化夹爪...")
    send_dashboard_command("stop")
    time.sleep(0.5)
    send_dashboard_command("load dh_init.urp")
    time.sleep(1)
    send_dashboard_command("play")
    time.sleep(3)
    send_dashboard_command("stop")
    print("  夹爪已初始化")

    # 启动摄像头
    print("\n[3/3] 启动摄像头...")
    pipelines = start_realsense()
    pipe_main = pipelines[0]
    pipe_wrist = pipelines[1]

    # 确认视角
    if args.skip_preview:
        print("\n跳过预览 (主视角=D435I, 腕部=D405)")
    else:
        print("\n预览摄像头，按 Enter 确认，按 s 交换视角")
        while True:
            try:
                f1 = pipe_main["pipe"].wait_for_frames(timeout_ms=2000).get_color_frame()
                f2 = pipe_wrist["pipe"].wait_for_frames(timeout_ms=2000).get_color_frame()
                if f1 and f2:
                    img1 = np.asanyarray(f1.get_data())
                    img2 = np.asanyarray(f2.get_data())
                    cv2.putText(img1, "Main", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img2, "Wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Preview - Enter=confirm, S=swap", np.hstack([img1, img2]))
            except RuntimeError:
                pass
            key = cv2.waitKey(100) & 0xFF
            if key == 13:
                break
            elif key == ord("s"):
                pipe_main, pipe_wrist = pipe_wrist, pipe_main
                print(f"已交换 -> 主视角: {pipe_main['name']}, 腕部: {pipe_wrist['name']}")
        cv2.destroyAllWindows()
    active_pipes = [pipe_main, pipe_wrist]

    # 推理客户端
    client = XVLAClient(host=args.server_ip, port=args.server_port,
                        domain_id=args.domain_id, steps=10,
                        use_https=args.use_https)

    print(f"\n任务: {args.instruction}")
    print(f"服务器: {client.url}")
    print(f"最大步数: {args.max_steps}, 每次推理执行: {args.chunk_size} 步")
    print("=" * 60)
    input("按 Enter 开始执行...")

    action_queue = deque()
    infer_times = []
    grip_state = 1.0
    speed = 0.3
    acceleration = 0.3
    skip_count = 0

    try:
        for step in range(args.max_steps):
            # 检查 RTDE 连接
            if not rtde.is_connected():
                if not rtde.reconnect():
                    print("  RTDE 重连失败，等待 3 秒后重试...")
                    time.sleep(3)
                    continue

            # 摄像头实时预览 + 推理采集
            img_main, img_wrist = capture_images(active_pipes)
            if img_main is None:
                print("  摄像头采集失败，跳过")
                continue

            preview = np.hstack([
                cv2.cvtColor(img_main, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
            ])
            cv2.imshow("UR7e - Main+Wrist", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if not action_queue:
                try:
                    tcp = rtde.get_tcp()
                    proprio = tcp_to_proprio(tcp, grip_state)
                except Exception as e:
                    print(f"  RTDE 读取失败: {e}")
                    continue

                t0 = time.time()
                try:
                    actions = client.predict(img_main, img_wrist, proprio, args.instruction)
                    t1 = time.time()
                    infer_ms = (t1 - t0) * 1000
                    infer_times.append(infer_ms)

                    # 打印所有预测动作
                    print(f"\n  [推理 #{len(infer_times)}] {infer_ms:.0f}ms | 预测 {len(actions)} 步")
                    for i, a in enumerate(actions):
                        p, r, g = parse_action(a)
                        print(f"    动作[{i:2d}] pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] grip={'开' if g>0.5 else '闭'}")

                    action_queue.clear()
                    for a in actions[: args.chunk_size]:
                        action_queue.append(a)

                except Exception as e:
                    print(f"  推理失败: {e}")
                    continue

                # 逐步模式：推理完暂停，用户手动控制
                if args.step_mode:
                    print(f"\n  --- 逐步执行模式 ---")
                    print(f"  已加载 {len(action_queue)} 步动作")
                    print(f"  Enter=执行下一步, s=跳过, q=退出\n")

            # 取一条动作执行
            action_20d = action_queue.popleft()
            pos, rotvec, grip_logit = parse_action(action_20d)
            target_tcp = np.concatenate([pos, rotvec]).tolist()

            # 逐步模式：等待用户确认
            if args.step_mode:
                print(f"  → 待执行: pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                      f"grip={'开' if grip_logit>0.5 else '闭'} "
                      f"队列剩余={len(action_queue)}")
                cmd = input("  命令 (Enter=执行/s=跳过/q=退出): ").strip().lower()
                if cmd == 'q':
                    break
                elif cmd == 's':
                    print("  已跳过")
                    continue

            # 安全检查
            try:
                current_tcp = rtde.get_tcp()
            except Exception as e:
                print(f"  RTDE 读取失败: {e}")
                action_queue.clear()
                continue

            if not check_action_safety(current_tcp, target_tcp):
                skip_count += 1
                continue

            # 夹爪
            if grip_logit > 0.5:
                grip_state = 1.0
                grip_pos = 100.0
            else:
                grip_state = 0.0
                grip_pos = 0.0

            # 发送 moveL + 夹爪
            try:
                rtde.moveL(target_tcp, speed, acceleration, grip_pos=grip_pos)
            except Exception as e:
                print(f"  moveL 失败: {e}")
                action_queue.clear()
                if not rtde.reconnect():
                    time.sleep(1)
                continue

            if step % 5 == 0:
                print(f"  步骤 {step}/{args.max_steps} | "
                      f"pos={[f'{v:.3f}' for v in pos]} | "
                      f"grip={'开' if grip_state > 0.5 else '闭'} | "
                      f"队列={len(action_queue)} | "
                      f"跳过={skip_count}")

            # 控制频率节拍
            if not args.step_mode:
                time.sleep(CONTROL_DT)

    except KeyboardInterrupt:
        print("\n用户中断")

    # 汇总
    print("=" * 60)
    if infer_times:
        print(f"推理次数: {len(infer_times)}, "
              f"平均 {np.mean(infer_times):.0f}ms, "
              f"最小 {np.min(infer_times):.0f}ms, "
              f"最大 {np.max(infer_times):.0f}ms")
    print(f"跳过动作: {skip_count} 步")

    # 清理
    for p in active_pipes:
        p["pipe"].stop()
    cv2.destroyAllWindows()
    print("结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", default="localhost")
    parser.add_argument("--server_port", type=int, default=8010)
    parser.add_argument("--domain_id", type=int, default=12)
    parser.add_argument("--instruction", default="Pick up the chili on the table")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--use_https", action="store_true", default=False)
    parser.add_argument("--skip_preview", action="store_true", default=False,
                        help="跳过摄像头预览")
    parser.add_argument("--step_mode", action="store_true", default=False,
                        help="单次推理 + 逐步执行模式（按 Enter 执行每一步）")
    run(parser.parse_args())
