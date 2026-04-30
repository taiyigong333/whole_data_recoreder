"""
UR7e + X-VLA 仿真客户端（含 IK 控制 + 底座坐标变换 + Action Chunking）
"""

import argparse, json, os, time, signal
import numpy as np
import json_numpy, requests, mujoco
from scipy.spatial.transform import Rotation as R
from collections import deque

HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# 全局变量，Ctrl+C 时能保存视频
_frames = []
_save_video = False
_output_dir = "logs"

def _save_on_exit(signum, frame):
    if _save_video and _frames:
        os.makedirs(_output_dir, exist_ok=True)
        import imageio
        video_path = os.path.join(_output_dir, "ur7e_rollout.mp4")
        imageio.mimsave(video_path, _frames, fps=30)
        print(f"\n[Ctrl+C] Video saved: {video_path} ({len(_frames)} frames)")
    exit(0)

signal.signal(signal.SIGINT, _save_on_exit)

def euler_to_rotate6d(euler, pattern="xyz"):
    return R.from_euler(pattern, euler, degrees=False).as_matrix()[..., :, :2].reshape(euler.shape[:-1] + (6,))

def rotate6d_to_axis_angle(v6):
    v6 = np.asarray(v6)
    a1 = v6[..., 0:5:2]; a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return R.from_matrix(np.stack((b1, b2, b3), axis=-1)).as_rotvec()

class UR7eMuJoCoEnv:
    def __init__(self, scene_xml_path):
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME_QPOS
        self.data.ctrl[:6] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

    def get_image(self, camera_name="front"):
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def _get_base_transform(self):
        base_pos = self.data.xpos[self.base_body_id].copy()
        base_mat = self.data.xmat[self.base_body_id].reshape(3, 3).copy()
        return base_pos, base_mat

    def get_ee_pose_world(self):
        pos = self.data.site_xpos[self.ee_site_id].copy()
        mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return pos, mat

    def get_ee_pose_base(self):
        base_pos, base_mat = self._get_base_transform()
        pos_w, mat_w = self.get_ee_pose_world()
        pos_b = base_mat.T @ (pos_w - base_pos)
        mat_b = base_mat.T @ mat_w
        return pos_b, mat_b

    def get_proprio_xvla(self):
        pos_b, mat_b = self.get_ee_pose_base()
        rot6d = mat_b[:, :2].flatten()
        grip = np.array([1.0])
        left = np.concatenate([pos_b, rot6d, grip])
        right = np.zeros(10)
        return np.concatenate([left, right]).astype(np.float32)

    def base_to_world_pos(self, pos_b):
        base_pos, base_mat = self._get_base_transform()
        return base_mat @ pos_b + base_pos

    def base_to_world_mat(self, mat_b):
        _, base_mat = self._get_base_transform()
        return base_mat @ mat_b

    def solve_ik(self, target_pos_world, target_rotmat_world, max_iter=50, tol=0.005):
        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            current_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            pos_err = target_pos_world - current_pos
            rot_err_mat = target_rotmat_world @ current_mat.T
            rot_err = R.from_matrix(rot_err_mat).as_rotvec()
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            jacp = jacp[:, :6]; jacr = jacr[:, :6]
            jac = np.vstack([jacp, jacr])
            damping = 0.05
            delta_q = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
            max_step = 0.1
            scale = max(1.0, np.max(np.abs(delta_q)) / max_step)
            delta_q /= scale
            self.data.qpos[:6] += delta_q

    def step(self, action_7d_world):
        target_pos = action_7d_world[:3]
        target_rot = R.from_rotvec(action_7d_world[3:6]).as_matrix()
        grip = action_7d_world[6]
        self.solve_ik(target_pos, target_rot)
        self.data.ctrl[:6] = self.data.qpos[:6]
        self.data.ctrl[6] = 0.0 if grip > 0.5 else 255.0
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)


class XVLAClient:
    def __init__(self, host="localhost", port=8000, domain_id=12, steps=10):
        self.url = f"http://{host}:{port}/act"
        self.domain_id = domain_id
        self.steps = steps

    def predict(self, image, proprio, instruction):
        payload = {
            "proprio": json_numpy.dumps(proprio),
            "language_instruction": instruction,
            "image0": json_numpy.dumps(image),
            "domain_id": self.domain_id,
            "steps": self.steps,
        }
        response = requests.post(self.url, json=payload, timeout=30)
        response.raise_for_status()
        return np.array(response.json()["action"], dtype=np.float32)

    def parse_action(self, action_20d):
        left = action_20d[:10]
        pos_base = left[:3]
        rot6d_base = left[3:9]
        grip_logit = left[9]
        axis_angle_base = rotate6d_to_axis_angle(rot6d_base)
        mat_base = R.from_rotvec(axis_angle_base).as_matrix()
        return pos_base, mat_base, grip_logit


def run_evaluation(args):
    global _frames, _save_video, _output_dir

    env = UR7eMuJoCoEnv(args.scene_xml)
    client = XVLAClient(host=args.server_ip, port=args.server_port,
                        domain_id=args.domain_id, steps=10)
    instruction = args.instruction
    max_steps = args.max_steps
    chunk_size = args.chunk_size
    frames = []

    _save_video = args.save_video
    _output_dir = args.output_dir

    action_queue = deque()

    print(f"Task: {instruction}")
    print(f"Max steps: {max_steps}, Chunk size: {chunk_size}")
    print(f"Domain ID: {args.domain_id}")
    print("-" * 60)

    env.reset()

    infer_times = []
    t_total_start = time.time()

    for step in range(max_steps):
        # action chunking: 队列空了才重新推理
        if not action_queue:
            image = env.get_image(camera_name="front")
            proprio = env.get_proprio_xvla()

            if args.save_video:
                frames.append(image.copy())
                _frames = frames

            t0 = time.time()
            actions = client.predict(image, proprio, instruction)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000
            infer_times.append(infer_ms)

            for a in actions[:chunk_size]:
                action_queue.append(a)

            print(f"  [Infer #{len(infer_times)}] {infer_ms:.0f}ms | "
                  f"queued {min(chunk_size, len(actions))} actions")

        action_20d = action_queue.popleft()
        pos_base, mat_base, grip_logit = client.parse_action(action_20d)

        target_pos_world = env.base_to_world_pos(pos_base)
        target_mat_world = env.base_to_world_mat(mat_base)
        target_rotvec_world = R.from_matrix(target_mat_world).as_rotvec()
        grip = 1.0 if grip_logit > 0.5 else -1.0
        action_7d_world = np.concatenate([target_pos_world, target_rotvec_world, [grip]])

        env.step(action_7d_world)

        if step % 10 == 0:
            print(f"  Step {step}/{max_steps} | "
                  f"pos_world={target_pos_world.round(3)} | "
                  f"grip={grip:.2f} | queue={len(action_queue)}")

    t_total = time.time() - t_total_start
    print("-" * 60)
    print(f"Complete. {max_steps} steps in {t_total:.1f}s ({max_steps/t_total:.1f} FPS)")
    if infer_times:
        print(f"Inference calls: {len(infer_times)}, "
              f"avg {np.mean(infer_times):.0f}ms, "
              f"min {np.min(infer_times):.0f}ms, "
              f"max {np.max(infer_times):.0f}ms")

    if args.save_video and frames:
        os.makedirs(args.output_dir, exist_ok=True)
        import imageio
        video_path = os.path.join(args.output_dir, "ur7e_rollout.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_xml", default="ur7e_scene.xml")
    parser.add_argument("--server_ip", default="localhost")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--domain_id", type=int, default=12)
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="每次推理执行的步数 (1-30, 越大越快但精度越低)")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--save_video", action="store_true", default=True)
    run_evaluation(parser.parse_args())
