"""
DH AG 夹爪诊断脚本
分步验证：TCP 连通 → URScript 执行 → URCap 函数有效性 → 夹爪状态
"""

import socket
import time

ROBOT_IP = "192.168.1.88"


def send_ur(script, port=30002):
    with socket.create_connection((ROBOT_IP, port), timeout=5) as s:
        s.sendall(script.encode("utf-8"))


def main():
    print("DH AG 夹爪诊断工具")
    print(f"目标: {ROBOT_IP}:30002")
    print("=" * 50)
    print("提示：示教器上打开 I/O → 数字输出(Standard) 面板")
    print("      本脚本会用 DO7 测试连通性，用 DO0/DO1 传递夹爪状态")
    print("=" * 50)

    # ============ 第1步：TCP 连通性 ============
    print("\n[第1步] TCP 30002 连通性")
    try:
        s = socket.create_connection((ROBOT_IP, 30002), timeout=5)
        s.close()
        print("  ✓ 连接成功")
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return

    # ============ 第2步：URScript 能否执行 ============
    print("\n[第2步] URScript 执行测试")
    print("  发送: set_standard_digital_out(7, True)")

    send_ur("set_standard_digital_out(7, False)\n")
    time.sleep(0.5)
    send_ur("set_standard_digital_out(7, True)\n")
    time.sleep(1)

    ans = input("  示教器 I/O 面板上 DO7 是否为 ON？(y/n): ").strip().lower()

    if ans != 'y':
        print("  ✗ DO7 没有 ON → 30002 的 URScript 没有被执行")
        print("\n  尝试端口 30003...")
        try:
            send_ur("set_standard_digital_out(7, True)\n", port=30003)
            time.sleep(1)
            ans2 = input("  用 30003 端口，DO7 变 ON 了吗？(y/n): ").strip().lower()
            if ans2 == 'y':
                print("  ✓ 30003 可以执行！后续请把代码里的 30002 都改成 30003")
            else:
                print("  ✗ 都不行，排查：")
                print("    - 控制器是否在远程控制模式？")
                print("    - 示教器上是否有程序正在运行？先停止")
                print("    - 尝试重启控制柜")
            return
        except Exception as e:
            print(f"  ✗ 30003 连接失败: {e}")
            return

    print("  ✓ URScript 可以执行！")
    send_ur("set_standard_digital_out(7, False)\n")

    # ============ 第3步：测试不同 URCap 函数前缀 ============
    print("\n[第3步] 测试 DH URCap 函数名")
    print("  脚本会把夹爪状态写到 DO0(is_connected) 和 DO1(is_activated)")

    prefixes = ["dh_ag95", "dh_ag"]

    for prefix in prefixes:
        print(f"\n  --- 尝试前缀: {prefix}_* ---")

        send_ur("set_standard_digital_out(0, False)\n"
                "set_standard_digital_out(1, False)\n")
        time.sleep(0.5)

        script = (
            f"def diag():\n"
            f"  global c = {prefix}_is_connected(1)\n"
            f"  global a = {prefix}_is_activated(1)\n"
            f"  set_standard_digital_out(0, c)\n"
            f"  set_standard_digital_out(1, a)\n"
            f"  textmsg(\"DIAG: {prefix} conn=\" + to_str(c) + \" act=\" + to_str(a))\n"
            f"end\ndiag()\n"
        )
        try:
            send_ur(script)
            time.sleep(2)

            print(f"  请看示教器 I/O 面板 DO0 和 DO1 的状态")
            ans = input(f"  DO0,DO1 是什么？(如 1,1 或 0,0): ").strip().replace(" ", "")

            if ans.startswith("1"):
                print(f"  ✓ 前缀 {prefix} 有效！夹爪已连接")
                if ans == "1,1":
                    print("  ✓ 夹爪已激活！")
                else:
                    print("  ⚠ 夹爪已连接但未激活，请在示教器上激活夹爪")
                print(f"\n{'=' * 50}")
                print(f"结论：函数前缀 = {prefix}")
                print(f"请把 collect_freedrive.py 和 test_gripper.py 中的")
                print(f"GRIPPER_MODEL 改为 \"{prefix.replace('dh_', '')}\"")
                print(f"{'=' * 50}")
                break
            elif ans == "0,0":
                print(f"  ✗ DO0=0 → 夹爪未连接或前缀 {prefix} 无效")
            else:
                print(f"  ? 不确定，可以在示教器日志里搜 'DIAG' 看详细输出")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    else:
        print(f"\n  ✗ 所有前缀都没找到连接的夹爪")
        print("  可能原因：")
        print("    - URCap 里没有正确配置夹爪的连接模式和设备地址")
        print("    - RS485 线没接好")
        print("    - 夹爪型号不匹配这个 URCap 插件")
        print("  建议：在示教器 程序编辑器 里找 DH 相关命令，看函数名是什么")

    # 清理 DO
    send_ur("set_standard_digital_out(0, False)\n"
            "set_standard_digital_out(1, False)\n")

    print("\n诊断结束。")


if __name__ == "__main__":
    main()
