from rtde_receive import RTDEReceiveInterface

rtde_r = RTDEReceiveInterface("192.168.1.88")

tcp = rtde_r.getActualTCPPose()
q = rtde_r.getActualQ()

print(f"末端位姿 [x, y, z, rx, ry, rz]: {tcp}")
print(f"关节角度: {q}")
print("连接成功！")