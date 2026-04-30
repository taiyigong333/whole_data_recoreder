import numpy as np
from pathlib import Path

src = Path(r"E:\data_fetch\XVLA-Code\raw_demos\episode_0000.npz")
dst = Path(r"E:\data_fetch\XVLA-Code\raw_demos_10hz\episode_0000_10hz.npz")

dst.parent.mkdir(parents=True, exist_ok=True)

data = np.load(src, allow_pickle=True)

stride = 3  # 30Hz -> 10Hz
new_data = {}

for key in data.files:
    arr = data[key]

    if key in ["images", "images_wrist", "tcp_poses", "gripper"]:
        new_data[key] = arr[::stride]
    elif key == "fps":
        new_data[key] = np.array(10)
    else:
        new_data[key] = arr

np.savez_compressed(dst, **new_data)


print("saved:", dst)
print("original frames:", data["images"].shape[0])
print("new frames:", new_data["images"].shape[0])
print("new fps:", new_data["fps"])