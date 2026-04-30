"""
DH AG 夹爪控制（寄存器桥接方案）
通过 Modbus TCP 写 UR input 寄存器下指令
示教器上需运行桥接程序（read_input_integer_register(0) 监听命令）

寄存器映射：
  Input R0:  命令 (Python→UR)  1=打开, 2=闭合
"""

import time
from pymodbus.client import ModbusTcpClient

ROBOT_IP = "192.168.1.88"

# UR input integer register 0 对应 Modbus holding register 128
REG_CMD_ADDRESS = 128


def main():
    print("连接 UR Modbus TCP...")
    client = ModbusTcpClient(ROBOT_IP, port=502)
    if not client.connect():
        print("✗ 连接失败，检查网络")
        return
    print("✓ 连接成功")
    print("=" * 50)
    print("  c = 闭合    g = 打开    q = 退出")
    print("=" * 50)

    while True:
        key = input("\n命令: ").strip().lower()

        if key == 'g':
            result = client.write_register(REG_CMD_ADDRESS, 1)
            if result.isError():
                print("  ✗ 写入失败")
            else:
                print("  → 夹爪打开")
            time.sleep(1)

        elif key == 'c':
            result = client.write_register(REG_CMD_ADDRESS, 2)
            if result.isError():
                print("  ✗ 写入失败")
            else:
                print("  → 夹爪闭合")
            time.sleep(1)

        elif key == 'q':
            print("退出")
            break
        else:
            print("  未知命令")

    client.close()


if __name__ == "__main__":
    main()
