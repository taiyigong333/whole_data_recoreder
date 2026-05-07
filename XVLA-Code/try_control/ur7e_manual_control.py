"""
UR7e 手动控制脚本

从 ur7e_real_client.py 中抽出真实控制部分，支持两种模式：
1. 交互式手动输入
2. 命令行一次性执行

支持能力：
- 机械臂 TCP 绝对位姿
- 机械臂 TCP 相对增量
- 夹爪开合或指定位置
"""

import argparse
import math
import shlex
import socket
import threading
import time

from xmlrpc.server import SimpleXMLRPCServer


DASHBOARD_PORT = 29999
DEFAULT_ROBOT_IP = "192.168.1.88"
DEFAULT_GRIPPER_PORT = 50000
DEFAULT_CONTROL_HZ = 5
DEFAULT_MAX_POS_DELTA = 0.10
DEFAULT_MAX_ROT_DELTA = 0.50


def clip(value, low, high):
    return max(low, min(high, value))


def l2_norm(values):
    return math.sqrt(sum(value * value for value in values))


class GripperXMLRPCBridge:
    def __init__(self, robot_ip, port=DEFAULT_GRIPPER_PORT):
        self.robot_ip = robot_ip
        self.port = port
        self.target = 100.0
        self.started = False
        self._server_thread = None

    def get_pos(self):
        return float(self.target)

    def _run_server(self):
        server = SimpleXMLRPCServer(
            ("0.0.0.0", self.port),
            allow_none=True,
            logRequests=False,
        )
        server.register_function(self.get_pos, "get_pos")
        server.serve_forever()

    def _send_dashboard_command(self, command):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            sock.connect((self.robot_ip, DASHBOARD_PORT))
            sock.recv(1024)
            if not command.endswith("\n"):
                command += "\n"
            sock.sendall(command.encode("utf-8"))
            sock.recv(1024)

    def start(self, program_name):
        self.target = 100.0
        if self._server_thread is None:
            self._server_thread = threading.Thread(target=self._run_server, daemon=True)
            self._server_thread.start()
            time.sleep(0.5)

        self._send_dashboard_command(f"load {program_name}")
        time.sleep(1)
        self._send_dashboard_command("play")
        self.started = True
        print(f"  夹爪服务已启动: {program_name} (XML-RPC 端口 {self.port})")

    def set_position(self, position):
        clipped = float(clip(position, 0.0, 100.0))
        self.target = clipped
        print(f"  夹爪目标位置: {clipped:.1f} (0=闭合, 100=张开)")

    def open(self):
        self.set_position(100.0)

    def close(self):
        self.set_position(0.0)


class RTDEConnection:
    def __init__(self, ip):
        self.ip = ip
        self.ctrl = None
        self.recv = None
        self.connect()

    def connect(self):
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ImportError(
                "缺少 RTDE 依赖，请先安装 rtde_control / rtde_receive 对应 Python 包"
            ) from exc

        print(f"  连接 RTDE: {self.ip} ...")
        self.ctrl = RTDEControlInterface(self.ip)
        self.recv = RTDEReceiveInterface(self.ip)
        tcp = self.recv.getActualTCPPose()
        print(f"  RTDE 已连接, TCP: {[f'{v:.4f}' for v in tcp]}")

    def is_connected(self):
        try:
            self.recv.getActualTCPPose()
            return True
        except Exception:
            return False

    def reconnect(self):
        print("  RTDE 连接断开，尝试重连...")
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface

            self.ctrl = RTDEControlInterface(self.ip)
            self.recv = RTDEReceiveInterface(self.ip)
            tcp = self.recv.getActualTCPPose()
            print(f"  RTDE 重连成功, TCP: {[f'{v:.4f}' for v in tcp]}")
            return True
        except Exception as exc:
            print(f"  RTDE 重连失败: {exc}")
            return False

    def get_tcp(self):
        return self.recv.getActualTCPPose()

    def moveL(self, target, speed, accel):
        return self.ctrl.moveL(target, speed, accel)

    def stop(self):
        try:
            self.ctrl.stopScript()
        except Exception:
            pass


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


class UR7eManualController:
    def __init__(
        self,
        robot_ip,
        control_hz,
        max_pos_delta,
        max_rot_delta,
        enable_gripper=True,
        gripper_program="test_xmlrpc.urp",
        gripper_port=DEFAULT_GRIPPER_PORT,
    ):
        self.control_dt = 1.0 / control_hz
        self.max_pos_delta = max_pos_delta
        self.max_rot_delta = max_rot_delta
        self.gripper_program = gripper_program

        self.rtde = RTDEConnection(robot_ip)
        self.gripper = (
            GripperXMLRPCBridge(robot_ip, gripper_port) if enable_gripper else None
        )

    def start(self):
        if self.gripper is None:
            print("  已跳过夹爪初始化")
            return
        self.gripper.start(self.gripper_program)

    def ensure_connected(self):
        if self.rtde.is_connected():
            return True
        return self.rtde.reconnect()

    def print_status(self):
        tcp = self.rtde.get_tcp()
        tcp_fmt = ", ".join(f"{value:.4f}" for value in tcp)
        print(f"  当前 TCP: [{tcp_fmt}]")
        if self.gripper is not None:
            if self.gripper.started:
                print(f"  当前夹爪目标: {self.gripper.target:.1f}")
            else:
                print("  当前夹爪: 未启动")

    def move_absolute(self, target_tcp, speed, acceleration, unsafe=False):
        if len(target_tcp) != 6:
            raise ValueError("target_tcp 必须包含 6 个数: x y z rx ry rz")

        if not self.ensure_connected():
            raise RuntimeError("RTDE 未连接，无法执行 moveL")

        current_tcp = self.rtde.get_tcp()
        is_safe, reason = check_action_safety(
            current_tcp,
            target_tcp,
            self.max_pos_delta,
            self.max_rot_delta,
        )
        if not is_safe and not unsafe:
            print(f"  已取消移动: {reason}")
            print("  如果确认安全，可重新运行脚本并加上 --unsafe")
            return False

        print(f"  执行 moveL -> {[round(v, 4) for v in target_tcp]}")
        self.rtde.moveL(target_tcp, speed, acceleration)
        time.sleep(self.control_dt)
        self.print_status()
        return True

    def move_delta(self, delta_tcp, speed, acceleration, unsafe=False):
        if len(delta_tcp) != 6:
            raise ValueError("delta_tcp 必须包含 6 个数: dx dy dz drx dry drz")

        if not self.ensure_connected():
            raise RuntimeError("RTDE 未连接，无法执行 moveL")

        current_tcp = self.rtde.get_tcp()
        target_tcp = [
            current_tcp[index] + delta_tcp[index] for index in range(6)
        ]
        print(f"  相对移动 delta -> {[round(v, 4) for v in delta_tcp]}")
        return self.move_absolute(target_tcp, speed, acceleration, unsafe=unsafe)

    def set_gripper(self, position):
        if self.gripper is None:
            print("  当前脚本未启用夹爪控制")
            return
        self.gripper.set_position(position)

    def open_gripper(self):
        self.set_gripper(100.0)

    def close_gripper(self):
        self.set_gripper(0.0)

    def shutdown(self):
        self.rtde.stop()


def print_help():
    print("\n可用命令:")
    print("  status")
    print("    查看当前 TCP 位姿和夹爪目标")
    print("  pose x y z rx ry rz")
    print("    机械臂移动到绝对 TCP 位姿")
    print("  delta dx dy dz drx dry drz")
    print("    机械臂按当前位姿做相对移动")
    print("  open")
    print("    夹爪张开 (100)")
    print("  close")
    print("    夹爪闭合 (0)")
    print("  grip pos")
    print("    设置夹爪位置，范围 0~100")
    print("  help")
    print("    打印帮助")
    print("  quit / exit")
    print("    退出脚本")


def has_gripper_action(args):
    return args.open or args.close or args.grip is not None


def has_one_shot_action(args):
    return args.pose is not None or args.delta is not None or has_gripper_action(args) or args.status


def run_one_shot(args, controller):
    if has_gripper_action(args) and controller.gripper is not None:
        controller.start()

    controller.print_status()

    if args.pose is not None:
        controller.move_absolute(
            list(args.pose),
            speed=args.speed,
            acceleration=args.acceleration,
            unsafe=args.unsafe,
        )
    elif args.delta is not None:
        controller.move_delta(
            list(args.delta),
            speed=args.speed,
            acceleration=args.acceleration,
            unsafe=args.unsafe,
        )

    if args.open:
        controller.open_gripper()
    elif args.close:
        controller.close_gripper()
    elif args.grip is not None:
        controller.set_gripper(args.grip)

    if args.wait_after > 0 and (
        args.pose is not None
        or args.delta is not None
        or has_gripper_action(args)
    ):
        time.sleep(args.wait_after)

    if args.pose is not None or args.delta is not None or has_gripper_action(args):
        print("\n执行完成")
        controller.print_status()


def run_interactive_cli(args, controller):
    if not args.no_gripper:
        controller.start()

    controller.print_status()
    print_help()

    while True:
        raw = input("\n命令> ").strip()
        if not raw:
            continue

        parts = shlex.split(raw)
        command = parts[0].lower()

        try:
            if command in {"quit", "exit", "q"}:
                print("退出控制")
                break

            if command == "help":
                print_help()
                continue

            if command == "status":
                controller.print_status()
                continue

            if command == "pose":
                if len(parts) != 7:
                    print("  用法: pose x y z rx ry rz")
                    continue
                target_tcp = [float(value) for value in parts[1:7]]
                controller.move_absolute(
                    target_tcp,
                    speed=args.speed,
                    acceleration=args.acceleration,
                    unsafe=args.unsafe,
                )
                continue

            if command == "delta":
                if len(parts) != 7:
                    print("  用法: delta dx dy dz drx dry drz")
                    continue
                delta_tcp = [float(value) for value in parts[1:7]]
                controller.move_delta(
                    delta_tcp,
                    speed=args.speed,
                    acceleration=args.acceleration,
                    unsafe=args.unsafe,
                )
                continue

            if command == "open":
                controller.open_gripper()
                continue

            if command == "close":
                controller.close_gripper()
                continue

            if command == "grip":
                if len(parts) != 2:
                    print("  用法: grip pos")
                    continue
                controller.set_gripper(float(parts[1]))
                continue

            print("  未知命令，输入 help 查看帮助")
        except ValueError as exc:
            print(f"  参数错误: {exc}")
        except Exception as exc:
            print(f"  执行失败: {exc}")


def run_manual_cli(args):
    print("=" * 60)
    print("UR7e 手动控制")
    print(f"机器人 IP: {args.robot_ip}")
    print(f"速度: {args.speed:.3f} | 加速度: {args.acceleration:.3f}")
    print(
        f"安全阈值: 位移 {args.max_pos_delta * 100:.1f}cm | "
        f"旋转 {args.max_rot_delta:.3f}rad"
    )
    print("=" * 60)

    controller = UR7eManualController(
        robot_ip=args.robot_ip,
        control_hz=args.control_hz,
        max_pos_delta=args.max_pos_delta,
        max_rot_delta=args.max_rot_delta,
        enable_gripper=not args.no_gripper,
        gripper_program=args.gripper_program,
        gripper_port=args.gripper_port,
    )

    try:
        if has_one_shot_action(args):
            run_one_shot(args, controller)
        else:
            run_interactive_cli(args, controller)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        controller.shutdown()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default=DEFAULT_ROBOT_IP, help="UR 机器人 IP")
    parser.add_argument("--speed", type=float, default=0.05, help="moveL 速度")
    parser.add_argument("--acceleration", type=float, default=0.05, help="moveL 加速度")
    parser.add_argument("--control_hz", type=float, default=DEFAULT_CONTROL_HZ, help="控制节拍")
    parser.add_argument("--max_pos_delta", type=float, default=DEFAULT_MAX_POS_DELTA, help="单步最大位移")
    parser.add_argument("--max_rot_delta", type=float, default=DEFAULT_MAX_ROT_DELTA, help="单步最大旋转")
    parser.add_argument("--gripper_program", default="test_xmlrpc.urp", help="示教器上加载的夹爪程序")
    parser.add_argument("--gripper_port", type=int, default=DEFAULT_GRIPPER_PORT, help="本地 XML-RPC 端口")
    parser.add_argument("--no_gripper", action="store_true", default=False, help="禁用夹爪控制")
    parser.add_argument("--unsafe", action="store_true", default=False, help="跳过单步位姿安全阈值检查")
    parser.add_argument("--wait_after", type=float, default=0.2, help="一次性执行后额外等待秒数")

    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument(
        "--pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="一次性执行绝对 TCP 位姿移动",
    )
    motion_group.add_argument(
        "--delta",
        type=float,
        nargs=6,
        metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ"),
        help="一次性执行相对 TCP 位姿移动",
    )

    grip_group = parser.add_mutually_exclusive_group()
    grip_group.add_argument("--grip", type=float, help="一次性设置夹爪位置，范围 0~100")
    grip_group.add_argument("--open", action="store_true", default=False, help="一次性张开夹爪")
    grip_group.add_argument("--close", action="store_true", default=False, help="一次性闭合夹爪")

    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="打印当前状态后退出；若和动作参数一起提供，则执行前后都会打印状态",
    )
    return parser


if __name__ == "__main__":
    run_manual_cli(build_parser().parse_args())

# python ur7e_manual_control.py --status
# python ur7e_manual_control.py --delta 0 0 0.02 0 0 0 --status
# python ur7e_manual_control.py --pose 0.35 -0.20 0.25 2.22 -2.22 0.00
# python ur7e_manual_control.py --open
# python ur7e_manual_control.py --grip 30
# python ur7e_manual_control.py --delta 0 0 0.01 0 0 0 --no_gripper
