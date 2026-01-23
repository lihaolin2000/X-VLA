import json_numpy
import requests
import numpy as np
class XvlaClient:
    def __init__(self, 
                 server_url = "http://0.0.0.0:8000/act",
                 timeout = 50):
        self.server_url = server_url
        self.timeout = timeout

    def get_action(self, 
                   proprio = np.zeros(7, dtype=np.float32), 
                   image = np.zeros((256, 256, 3), dtype=np.uint8), 
                   instruction = "Move the gripper to the target position"):
        print(image['top_camera'].shape)
        payload = {
            "proprio": json_numpy.dumps(proprio),
            "language_instruction": instruction,
            "image0": json_numpy.dumps(image['top_camera']),
            "image1": json_numpy.dumps(image['left_camera']),
            "image2": json_numpy.dumps(image['right_camera']),
            "domain_id": 0,
            "steps": 10
        }

        try:
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            actions = np.array(result["action"], dtype=np.float32)
            print(f"✅ Received {actions.shape} predicted actions.")
        except Exception as e:
            print(f"⚠️ Request failed: {e}")
            actions = np.zeros((30, 20), dtype=np.float32)

        return actions

xvla_client = XvlaClient("http://10.140.66.146:8010/act")

import io, numpy as np, pyarrow.parquet as pq, av, cv2
from mmengine import fileio
from scipy.spatial.transform import Rotation as R

def read_bytes(path: str) -> bytes:
    return fileio.get(path)
def read_video_to_frames(path: str) -> np.ndarray:
    buf = io.BytesIO(read_bytes(path)); container = av.open(buf, options={'threads': '2'})
    frames = []
    for packet in container.demux(video=0):
        for f in packet.decode(): frames.append(f.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames, axis=0)


def read_parquet(path: str) -> dict:
    buf = io.BytesIO(read_bytes(path))
    return pq.read_table(buf).to_pydict()

image = {}
instruction = "put the bookmark on the top of the book."
data = read_parquet("/mnt/petrelfs/zhengjinliang/Data/sim_test/data/700-demo/task2/data/chunk-000/episode_000003.parquet")
image['top_camera'] = read_video_to_frames("/mnt/petrelfs/zhengjinliang/Data/sim_test/data/700-demo/task2/videos/chunk-000/video.top_camera_view/episode_000003.mp4")
image['left_camera'] = read_video_to_frames("/mnt/petrelfs/zhengjinliang/Data/sim_test/data/700-demo/task2/videos/chunk-000/video.left_camera_view/episode_000003.mp4")
image['right_camera'] = read_video_to_frames("/mnt/petrelfs/zhengjinliang/Data/sim_test/data/700-demo/task2/videos/chunk-000/video.right_camera_view/episode_000003.mp4")

pred_abs_joint = []
pred_base = []
gt_base = data['base.motion']
gt_abs_joint = data['action.joints']
gt_abs_joint = np.concatenate([gt_abs_joint[:1], gt_abs_joint[:-1]], axis=0)
print(image['top_camera'].shape)
print(gt_abs_joint.shape)
from tqdm import tqdm
for i in tqdm(range(0, len(gt_abs_joint), 30)):
    cur_image = {key: value[i] for key, value in image.items()}
    proprio = np.zeros(20, dtype=np.float32)
    proprio[:12] = gt_abs_joint[i]
    action = xvla_client.get_action(proprio, cur_image, instruction)
    for j in range(15):
        pred_abs_joint.append(action[j, :12] + proprio[:12])
        pred_base.append(action[j, 14:17])
        
# ===============================
# Stack predictions
# ===============================
pred_abs_joint = np.stack(pred_abs_joint, axis=0)   # (T, 12)
gt_abs_joint = gt_abs_joint[::2][:len(pred_abs_joint)]   # 对齐
pred_abs_joint = pred_abs_joint[:len(gt_abs_joint)]

print("pred_abs_joint:", pred_abs_joint.shape)
print("gt_abs_joint:", gt_abs_joint.shape)

import os
import numpy as np
import matplotlib.pyplot as plt

def save_all_dims_one_figure(pred, gt, save_path, prefix="joint"):
    """
    pred, gt: (T, D)
    """
    T, D = pred.shape
    t = np.arange(T)

    plt.figure(figsize=(12, 2.2 * D))

    for d in range(D):
        ax = plt.subplot(D, 1, d + 1)
        ax.plot(t, gt[:, d], label="GT")
        ax.plot(t, pred[:, d], label="Pred", linestyle="--")
        ax.set_ylabel(f"{prefix}_{d}")
        ax.grid(True)

        if d == 0:
            ax.legend()

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


save_all_dims_one_figure(
    pred_abs_joint,
    gt_abs_joint,
    save_path="abs_joint_pred_vs_gt_30_all_case2.png",
    prefix="abs_joint"
)


print("MAE per dim:", np.mean(np.abs(pred_abs_joint - gt_abs_joint), axis=0))
