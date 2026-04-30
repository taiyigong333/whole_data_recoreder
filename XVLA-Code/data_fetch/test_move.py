from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time

rtde_c = RTDEControlInterface("192.168.1.88")
rtde_r = RTDEReceiveInterface("192.168.1.88")

# 读取当前位姿
current = rtde_r.getActualTCPPose()
print(f"当前位置: x={current[0]:.3f}, y={current[1]:.3f}, z={current[2]:.3f}")

# 沿 Z 轴上移 2cm
target = current.copy()
target[2] += 0.02  # z + 2cm
print(f"目标位置: x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f}")

input("确认安全后按 Enter 执行移动（速度很慢）...")
rtde_c.moveL(target, speed=0.02, acceleration=0.02)  # 极慢：2cm/s

time.sleep(1)
new_pose = rtde_r.getActualTCPPose()
print(f"移动后: x={new_pose[0]:.3f}, y={new_pose[1]:.3f}, z={new_pose[2]:.3f}")

rtde_c.stopScript()