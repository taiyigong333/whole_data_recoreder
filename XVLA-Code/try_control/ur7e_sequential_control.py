"""
UR7e + DH AG 夹爪顺序控制脚本

按用户指定逻辑执行：
1. 先连接 RTDE，完成机械臂运动
2. 关闭 RTDE 相关连接
3. 再单独发送夹爪 URScript
4. 夹爪执行完成后关闭对应连接

用途：
- 机械臂单独控制
- 夹爪单独控制
- 机械臂和夹爪组合控制，但严格按“先臂后爪”的顺序执行
"""

import argparse
import math
import shlex
import socket
import threading
import time
from xmlrpc.server import SimpleXMLRPCServer


DEFAULT_ROBOT_IP = "192.168.1.88"
DEFAULT_SCRIPT_PORT = 30002
DEFAULT_DASHBOARD_PORT = 29999
DEFAULT_GRIPPER_PORT = 50000
DEFAULT_SPEED = 0.05
DEFAULT_ACCELERATION = 0.05
DEFAULT_MAX_POS_DELTA = 0.10
DEFAULT_MAX_ROT_DELTA = 0.50
DEFAULT_GRIPPER_PROGRAM = "test_xmlrpc.urp"
DEFAULT_GRIP_OPEN = 100.0
DEFAULT_GRIP_CLOSE = 0.0
DEFAULT_TRANSITION_DELAY = 0.3
DEFAULT_GRIPPER_WAIT = 2.0
DEFAULT_XMLRPC_CONNECT_TIMEOUT = 10.0
DEFAULT_XMLRPC_MIN_REQUESTS = 3


def clip(value, low, high):
    return max(low, min(high, value))


def l2_norm(values):
    return math.sqrt(sum(value * value for value in values))


def parse_grip_from_command(parts):
    if len(parts) == 8:
        return float(parts[7])
    return None


def has_motion_action(args):
    return args.pose is not None or args.delta is not None


def has_gripper_action(args):
    return args.grip is not None or args.open or args.close


def resolve_grip_target(args):
    if args.grip is not None:
        return float(args.grip)
    if args.open:
        return float(args.grip_open_position)
    if args.close:
        return float(args.grip_close_position)
    return None


def check_action_safety(current_tcp, target_tcp, max_pos_delta, max_rot_delta):
    delta_pos = l2_norm(
        target_tcp[index] - current_tcp[index] for index in range(3)
    )
    delta_rot = l2_norm(
        target_tcp[index] - current_tcp[index] for index in range(3, 6)
    )

    # if delta_pos > max_pos_delta:
    #     return False, f"位移过大: {delta_pos:.4f}m > {max_pos_delta:.4f}m"
    # if delta_rot > max_rot_delta:
    #     return False, f"旋转过大: {delta_rot:.4f}rad > {max_rot_delta:.4f}rad"
    return True, ""


class DashboardClient:
    def __init__(self, robot_ip, port=DEFAULT_DASHBOARD_PORT):
        self.robot_ip = robot_ip
        self.port = port

    def send(self, command):
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

    def stop_program(self):
        return self.send("stop")


class ArmRTDEController:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.ctrl = None
        self.recv = None
        self._connect()

    def _connect(self):
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ImportError(
                "缺少 RTDE 依赖，请安装 rtde_control 和 rtde_receive"
            ) from exc

        print(f"  连接 RTDE: {self.robot_ip} ...")
        self.ctrl = RTDEControlInterface(self.robot_ip)
        self.recv = RTDEReceiveInterface(self.robot_ip)
        tcp = self.recv.getActualTCPPose()
        print(f"  RTDE 已连接, TCP: {[f'{v:.4f}' for v in tcp]}")

    def get_tcp(self):
        return self.recv.getActualTCPPose()

    def print_status(self):
        tcp = self.get_tcp()
        tcp_fmt = ", ".join(f"{value:.4f}" for value in tcp)
        print(f"  当前 TCP: [{tcp_fmt}]")

    def move_absolute(self, target_tcp, speed, acceleration, unsafe, max_pos_delta, max_rot_delta):
        if len(target_tcp) != 6:
            raise ValueError("target_tcp 必须包含 6 个数: x y z rx ry rz")

        current_tcp = self.get_tcp()
        is_safe, reason = check_action_safety(
            current_tcp,
            target_tcp,
            max_pos_delta,
            max_rot_delta,
        )
        if not is_safe and not unsafe:
            raise RuntimeError(reason)

        print(f"  执行 moveL -> {[round(v, 4) for v in target_tcp]}")
        self.ctrl.moveL(target_tcp, speed, acceleration)

    def move_delta(self, delta_tcp, speed, acceleration, unsafe, max_pos_delta, max_rot_delta):
        if len(delta_tcp) != 6:
            raise ValueError("delta_tcp 必须包含 6 个数: dx dy dz drx dry drz")

        current_tcp = self.get_tcp()
        target_tcp = [current_tcp[index] + delta_tcp[index] for index in range(6)]
        print(f"  相对移动 delta -> {[round(v, 4) for v in delta_tcp]}")
        self.move_absolute(
            target_tcp,
            speed=speed,
            acceleration=acceleration,
            unsafe=unsafe,
            max_pos_delta=max_pos_delta,
            max_rot_delta=max_rot_delta,
        )

    def close(self):
        if self.ctrl is not None:
            try:
                self.ctrl.stopScript()
            except Exception:
                pass
            try:
                disconnect = getattr(self.ctrl, "disconnect", None)
                if callable(disconnect):
                    disconnect()
            except Exception:
                pass
            self.ctrl = None

        if self.recv is not None:
            try:
                disconnect = getattr(self.recv, "disconnect", None)
                if callable(disconnect):
                    disconnect()
            except Exception:
                pass
            self.recv = None

        print("  RTDE 连接已关闭")


class XMLRPCGripperController:
    def __init__(
        self,
        robot_ip,
        dashboard_port=DEFAULT_DASHBOARD_PORT,
        gripper_port=DEFAULT_GRIPPER_PORT,
        program_name=DEFAULT_GRIPPER_PROGRAM,
    ):
        self.robot_ip = robot_ip
        self.dashboard = DashboardClient(robot_ip, dashboard_port)
        self.gripper_port = gripper_port
        self.program_name = program_name
        self.target = DEFAULT_GRIP_OPEN
        self._server_thread = None
        self._request_event = threading.Event()
        self._request_count = 0

    def get_pos(self):
        self._request_count += 1
        self._request_event.set()
        return float(self.target)

    def _run_server(self):
        server = SimpleXMLRPCServer(
            ("0.0.0.0", self.gripper_port),
            allow_none=True,
            logRequests=False,
        )
        server.register_function(self.get_pos, "get_pos")
        server.serve_forever()

    def _ensure_server_started(self):
        if self._server_thread is not None:
            return
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        time.sleep(0.5)

    def _load_and_play_program(self):
        self.dashboard.send(f"load {self.program_name}")
        time.sleep(1.0)
        self.dashboard.send("play")

    def initialize(self, initial_target=None):
        self._ensure_server_started()
        self._request_event.clear()
        self._request_count = 0
        if initial_target is None:
            self.target = DEFAULT_GRIP_OPEN
        else:
            self.target = clip(float(initial_target), 0.0, 100.0)
        print(f"  启动夹爪程序: {self.program_name}")
        self._load_and_play_program()
        print(f"  XML-RPC 夹爪服务已启动 (端口 {self.gripper_port})")

    def move_to(self, position, stop_first=False, ensure_started=True):
        position = clip(float(position), 0.0, 100.0)

        if stop_first:
            try:
                reply = self.dashboard.stop_program()
                if reply:
                    print(f"  Dashboard stop: {reply}")
            except Exception as exc:
                print(f"  Dashboard stop 失败，继续发送夹爪脚本: {exc}")

        if ensure_started:
            self.initialize(initial_target=position)
        else:
            self.target = position

        self.target = position
        print(f"  设置夹爪目标 -> {position:.1f} (0=闭合, 100=张开)")

    def wait_for_robot_requests(self, timeout, min_requests):
        print(
            f"  等待机器人连接本地 XML-RPC 服务，"
            f"至少 {min_requests} 次请求，超时 {timeout:.1f}s ..."
        )
        ok = self._request_event.wait(timeout)
        if not ok:
            raise RuntimeError(
                f"机器人在 {timeout:.1f}s 内没有连上 XML-RPC 服务 "
                f"({self.robot_ip} -> 本机: {self.gripper_port})"
            )

        deadline = time.time() + timeout
        while self._request_count < min_requests and time.time() < deadline:
            time.sleep(0.05)

        if self._request_count < min_requests:
            raise RuntimeError(
                f"机器人已连上 XML-RPC，但请求次数只有 {self._request_count}，"
                f"未达到要求的 {min_requests}"
            )

        print(f"  机器人已连接 XML-RPC 服务，请求次数: {self._request_count}")


class SequentialURController:
    def __init__(self, args):
        self.args = args

    def print_status(self):
        arm = ArmRTDEController(self.args.robot_ip)
        try:
            arm.print_status()
        finally:
            arm.close()

    def run_motion_stage(self, target_pose=None, delta_pose=None):
        arm = ArmRTDEController(self.args.robot_ip)
        try:
            if target_pose is not None:
                arm.move_absolute(
                    target_pose,
                    speed=self.args.speed,
                    acceleration=self.args.acceleration,
                    unsafe=self.args.unsafe,
                    max_pos_delta=self.args.max_pos_delta,
                    max_rot_delta=self.args.max_rot_delta,
                )
            elif delta_pose is not None:
                arm.move_delta(
                    delta_pose,
                    speed=self.args.speed,
                    acceleration=self.args.acceleration,
                    unsafe=self.args.unsafe,
                    max_pos_delta=self.args.max_pos_delta,
                    max_rot_delta=self.args.max_rot_delta,
                )

            if self.args.motion_wait > 0:
                time.sleep(self.args.motion_wait)

            arm.print_status()
        finally:
            arm.close()

    def run_gripper_stage(self, grip_position, init_gripper=False):
        gripper = XMLRPCGripperController(
            robot_ip=self.args.robot_ip,
            dashboard_port=self.args.dashboard_port,
            gripper_port=self.args.gripper_port,
            program_name=self.args.gripper_program,
        )

        if init_gripper:
            gripper.initialize(initial_target=grip_position)
            if self.args.init_wait > 0:
                time.sleep(self.args.init_wait)

        gripper.move_to(
            grip_position,
            stop_first=self.args.stop_before_gripper,
            ensure_started=not init_gripper,
        )
        gripper.wait_for_robot_requests(
            self.args.xmlrpc_connect_timeout,
            self.args.xmlrpc_min_requests,
        )

        if self.args.gripper_wait > 0:
            time.sleep(self.args.gripper_wait)

        if self.args.stop_after_gripper:
            try:
                reply = gripper.dashboard.stop_program()
                if reply:
                    print(f"  Dashboard stop: {reply}")
            except Exception as exc:
                print(f"  停止夹爪程序失败: {exc}")

        print("  夹爪阶段完成")

    def run_sequence(self, target_pose=None, delta_pose=None, grip_position=None, init_gripper=False):
        if target_pose is not None or delta_pose is not None:
            print("\n[阶段 1] 机械臂运动")
            self.run_motion_stage(target_pose=target_pose, delta_pose=delta_pose)

        if grip_position is not None or init_gripper:
            if target_pose is not None or delta_pose is not None:
                print(f"\n等待 {self.args.transition_delay:.2f}s，准备切换到夹爪控制...")
                time.sleep(self.args.transition_delay)

            print("\n[阶段 2] 夹爪运动")
            if grip_position is None:
                raise ValueError("只初始化夹爪时请使用 init 命令或 --init_gripper")
            self.run_gripper_stage(grip_position, init_gripper=init_gripper)


def print_help(args):
    print("\n可用命令:")
    print("  status")
    print("    查看当前 TCP 位姿")
    print("  init")
    print("    初始化夹爪")
    print("  pose x y z rx ry rz [grip]")
    print("    先运动机械臂，再关闭 RTDE，再执行夹爪")
    print("  delta dx dy dz drx dry drz [grip]")
    print("    先运动机械臂，再关闭 RTDE，再执行夹爪")
    print("  grip pos")
    print(f"    单独控制夹爪，{args.grip_open_position:.0f}=张开, {args.grip_close_position:.0f}=闭合")
    print("  open")
    print("    单独张开夹爪")
    print("  close")
    print("    单独闭合夹爪")
    print("  help")
    print("    打印帮助")
    print("  quit / exit")
    print("    退出")


def run_one_shot(args, controller):
    grip_position = resolve_grip_target(args)

    if args.status and not has_motion_action(args) and grip_position is None and not args.init_gripper:
        controller.print_status()
        return

    if args.init_gripper and not has_motion_action(args) and grip_position is None:
        print("\n[阶段 1] 夹爪初始化")
        controller.run_gripper_stage(args.grip_open_position, init_gripper=True)
        return

    if args.pose is not None:
        controller.run_sequence(
            target_pose=list(args.pose),
            grip_position=grip_position,
            init_gripper=args.init_gripper,
        )
    elif args.delta is not None:
        controller.run_sequence(
            delta_pose=list(args.delta),
            grip_position=grip_position,
            init_gripper=args.init_gripper,
        )
    elif grip_position is not None:
        controller.run_gripper_stage(
            grip_position=grip_position,
            init_gripper=args.init_gripper,
        )


def run_interactive(args, controller):
    print_help(args)

    while True:
        raw = input("\n命令> ").strip()
        if not raw:
            continue

        parts = shlex.split(raw)
        command = parts[0].lower()

        try:
            if command in {"quit", "exit", "q"}:
                print("退出")
                break

            if command == "help":
                print_help(args)
                continue

            if command == "status":
                controller.print_status()
                continue

            if command == "init":
                controller.run_gripper_stage(args.grip_open_position, init_gripper=True)
                continue

            if command == "pose":
                if len(parts) not in {7, 8}:
                    print("  用法: pose x y z rx ry rz [grip]")
                    continue
                target_pose = [float(value) for value in parts[1:7]]
                grip_position = parse_grip_from_command(parts)
                controller.run_sequence(target_pose=target_pose, grip_position=grip_position)
                continue

            if command == "delta":
                if len(parts) not in {7, 8}:
                    print("  用法: delta dx dy dz drx dry drz [grip]")
                    continue
                delta_pose = [float(value) for value in parts[1:7]]
                grip_position = parse_grip_from_command(parts)
                controller.run_sequence(delta_pose=delta_pose, grip_position=grip_position)
                continue

            if command == "grip":
                if len(parts) != 2:
                    print("  用法: grip pos")
                    continue
                controller.run_gripper_stage(float(parts[1]))
                continue

            if command == "open":
                controller.run_gripper_stage(args.grip_open_position)
                continue

            if command == "close":
                controller.run_gripper_stage(args.grip_close_position)
                continue

            print("  未知命令，输入 help 查看帮助")
        except ValueError as exc:
            print(f"  参数错误: {exc}")
        except Exception as exc:
            print(f"  执行失败: {exc}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default=DEFAULT_ROBOT_IP, help="UR 机器人 IP")
    parser.add_argument("--dashboard_port", type=int, default=DEFAULT_DASHBOARD_PORT, help="Dashboard 端口")
    parser.add_argument("--gripper_port", type=int, default=DEFAULT_GRIPPER_PORT, help="本地 XML-RPC 夹爪端口")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="moveL 速度")
    parser.add_argument("--acceleration", type=float, default=DEFAULT_ACCELERATION, help="moveL 加速度")
    parser.add_argument("--max_pos_delta", type=float, default=DEFAULT_MAX_POS_DELTA, help="单步最大位移")
    parser.add_argument("--max_rot_delta", type=float, default=DEFAULT_MAX_ROT_DELTA, help="单步最大旋转")
    parser.add_argument("--unsafe", action="store_true", default=False, help="跳过运动安全阈值检查")
    parser.add_argument("--motion_wait", type=float, default=0.2, help="机械臂 moveL 后等待秒数")
    parser.add_argument("--transition_delay", type=float, default=DEFAULT_TRANSITION_DELAY, help="机械臂关闭后到夹爪开始前等待秒数")
    parser.add_argument("--gripper_wait", type=float, default=DEFAULT_GRIPPER_WAIT, help="夹爪脚本发送后等待秒数")
    parser.add_argument("--xmlrpc_connect_timeout", type=float, default=DEFAULT_XMLRPC_CONNECT_TIMEOUT, help="等待机器人连上本地 XML-RPC 服务的超时秒数")
    parser.add_argument("--xmlrpc_min_requests", type=int, default=DEFAULT_XMLRPC_MIN_REQUESTS, help="夹爪阶段要求机器人至少成功请求 XML-RPC 的次数")
    parser.add_argument("--init_wait", type=float, default=0.5, help="夹爪初始化后等待秒数")
    parser.add_argument("--stop_before_gripper", action="store_true", default=False, help="夹爪执行前发送 dashboard stop")
    parser.add_argument("--stop_after_gripper", action="store_true", default=True, help="夹爪完成后发送 dashboard stop，避免程序继续轮询 XML-RPC")

    parser.add_argument("--gripper_program", default=DEFAULT_GRIPPER_PROGRAM, help="示教器上用于 XML-RPC 控夹爪的程序名")
    parser.add_argument("--grip_open_position", type=float, default=DEFAULT_GRIP_OPEN, help="夹爪张开位置")
    parser.add_argument("--grip_close_position", type=float, default=DEFAULT_GRIP_CLOSE, help="夹爪闭合位置")
    parser.add_argument("--init_gripper", action="store_true", default=False, help="执行夹爪动作前先初始化夹爪")
    parser.add_argument("--status", action="store_true", default=False, help="查看当前 TCP 位姿")

    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument(
        "--pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="一次性绝对位姿控制",
    )
    motion_group.add_argument(
        "--delta",
        nargs=6,
        type=float,
        metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ"),
        help="一次性相对位姿控制",
    )

    grip_group = parser.add_mutually_exclusive_group()
    grip_group.add_argument("--grip", type=float, help="一次性设置夹爪位置值")
    grip_group.add_argument("--open", action="store_true", default=False, help="一次性张开夹爪")
    grip_group.add_argument("--close", action="store_true", default=False, help="一次性闭合夹爪")
    return parser


def main():
    args = build_parser().parse_args()
    controller = SequentialURController(args)

    print("=" * 60)
    print("UR7e 顺序控制")
    print(f"机器人 IP: {args.robot_ip}")
    print(f"moveL: speed={args.speed:.3f}, accel={args.acceleration:.3f}")
    print(
        f"夹爪程序: {args.gripper_program} | "
        f"open={args.grip_open_position:.0f}, close={args.grip_close_position:.0f}"
    )
    print("=" * 60)

    if args.pose is not None or args.delta is not None or has_gripper_action(args) or args.status or args.init_gripper:
        run_one_shot(args, controller)
    else:
        run_interactive(args, controller)


if __name__ == "__main__":
    main()
