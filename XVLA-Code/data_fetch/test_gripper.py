"""
DH AG 夹爪调试脚本（通过 URCap + TCP 30002）
用于在数据采集之前验证夹爪是否可控
"""

import socket
import time
import sys

# ============ 配置 ============
ROBOT_IP = "192.168.1.88"
GRIPPER_MODEL = "ag95"       # URCap 型号前缀（AG 系列统一用 ag95）
GRIPPER_INDEX = 1            # URCap 中的夹爪序号
GRIPPER_MODE = "TCI"         # 连接方式: USB=控制柜RS485, TCI=工具端, TCP=以太网
GRIPPER_DEVICE_ID = 2        # RS485 设备地址
GRIPPER_FORCE = 20           # 夹持力百分比 (20-100)
GRIPPER_SPEED = 20           # 夹持速度百分比 (1-100)
# ==============================


class DHGripperController:
    def __init__(self):
        self.robot_ip = ROBOT_IP
        self.prefix = f"dh_{GRIPPER_MODEL}"
        self.index = GRIPPER_INDEX
        self.mode = GRIPPER_MODE
        self.device_id = GRIPPER_DEVICE_ID
        self.force = GRIPPER_FORCE
        self.speed = GRIPPER_SPEED

    def _cmd(self, func, *args):
        args_str = ", ".join(
            f'"{a}"' if isinstance(a, str) else str(a) for a in args
        )
        return f"{self.prefix}_{func}({args_str})"

    def _send_program(self, lines):
        body = "\n".join(f"  {line}" for line in lines)
        script = f"def dh_job():\n{body}\nend\ndh_job()\n"
        print(f"  [发送] {script.strip()[:120]}...")
        with socket.create_connection((self.robot_ip, 30002), timeout=5.0) as sock:
            sock.sendall(script.encode("utf-8"))

    def test_connection(self):
        """第 1 步：测试 TCP 30002 是否可达"""
        print("\n[第1步] 测试 TCP 30002 连接...")
        try:
            sock = socket.create_connection((self.robot_ip, 30002), timeout=5.0)
            sock.close()
            print(f"  ✓ 成功连接 {self.robot_ip}:30002")
            return True
        except Exception as e:
            print(f"  ✗ 连接失败: {e}")
            print("  可能原因：")
            print("    - IP 地址不对（确认是 192.168.1.88）")
            print("    - 电脑和机器人不在同一网段")
            print("    - 控制柜未开机")
            return False

    def initialize(self):
        """第 2 步：初始化夹爪（扫描 + 连接 + 激活）"""
        print("\n[第2步] 初始化夹爪（约 8 秒）...")
        i, m, d = self.index, self.mode, self.device_id
        self._send_program([
            "sleep(3.0)",
            self._cmd("scan"),
            "sleep(3.0)",
            self._cmd("relate_device", i, m, d),
            "sleep(1.0)",
            self._cmd("connect", i),
            "sleep(1.0)",
            self._cmd("set_activate", i),
            self._cmd("wait_until_activated", i),
        ])
        print("  等待夹爪初始化完成...")
        time.sleep(9)
        print("  ✓ 初始化命令已发送")

    def open(self):
        """打开夹爪"""
        print("\n[打开夹爪]")
        self._send_program([
            self._cmd("set_speed", self.index, self.speed),
            self._cmd("set_position", self.index, 0.0),
        ])
        time.sleep(2)

    def close(self):
        """闭合夹爪"""
        print("\n[闭合夹爪]")
        self._send_program([
            self._cmd("set_force", self.index, self.force),
            self._cmd("set_speed", self.index, self.speed),
            self._cmd("set_position", self.index, 100.0),
        ])
        time.sleep(2)

    def move_to(self, position):
        """移动到指定位置 (0=全开, 100=全闭)"""
        print(f"\n[移动到位置 {position}]")
        self._send_program([
            self._cmd("set_force", self.index, self.force),
            self._cmd("set_speed", self.index, self.speed),
            self._cmd("set_position", self.index, float(position)),
        ])
        time.sleep(2)

    def interactive(self):
        """交互模式：键盘控制夹爪"""
        print("\n" + "=" * 50)
        print("交互控制模式")
        print("=" * 50)
        print("  c = 闭合    g = 打开    q = 退出")
        print("  0-9 = 位置百分比 (0=全开, 9=90%)")
        print("=" * 50)

        while True:
            key = input("\n输入命令: ").strip().lower()
            if key == 'c':
                self.close()
            elif key == 'g':
                self.open()
            elif key == 'q':
                print("退出")
                break
            elif key.isdigit():
                pos = int(key) * 10
                self.move_to(pos)
            else:
                print("  未知命令。c=闭合, g=打开, q=退出, 0-9=位置")


def main():
    print("DH AG 夹爪调试工具")
    print(f"目标: {ROBOT_IP}:30002 | 型号前缀: dh_{GRIPPER_MODEL}")
    print("=" * 50)

    gripper = DHGripperController()

    # 第 1 步：测试连接
    if not gripper.test_connection():
        print("\n连接失败，请检查网络后重试")
        return

    # 第 2 步：初始化
    try:
        gripper.initialize()
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        print("可能原因：")
        print("  - URCap 插件未安装或未激活")
        print("  - 示教器上夹爪未配置（检查 安装面板 → DH Grippers）")
        print("  - RS485 线未接好")
        print("  - 当前模式不允许外部脚本执行（尝试切换到远程控制模式）")
        return

    # 第 3 步：交互测试
    print("\n" + "=" * 50)
    print("初始化完成！下面进行交互测试")
    print("=" * 50)

    print("\n--- 测试 1: 打开夹爪 ---")
    gripper.open()
    resp = input("夹爪打开了吗？(y/n): ").strip().lower()

    if resp != 'y':
        print("夹爪没有反应，排查清单：")
        print("  1. 示教器上是否能看到 DH Grippers 配置？")
        print("  2. 示教器上手动操作夹爪能否动？")
        print("  3. RS485 线是否接在控制柜的正确端口？")
        print("  4. 设备地址(device_id)是否正确（默认 2）？")
        print("  5. 尝试在示教器上远程控制模式下运行")
        return

    print("\n--- 测试 2: 闭合夹爪 ---")
    gripper.close()
    resp = input("夹爪闭合了吗？(y/n): ").strip().lower()

    if resp != 'y':
        print("夹爪能打开但不能闭合，检查 force 和 speed 参数")
        return

    print("\n✓ 基础测试通过！")

    # 进入交互模式
    gripper.interactive()


if __name__ == "__main__":
    main()
