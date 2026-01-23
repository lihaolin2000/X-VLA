from datasets.utils import read_video_to_frames
from evaluation.websocket_client import ActionBuffer, HttpClient
import numpy as np
import json, time, os

def p(msg):  # tiny logger
    print(f"[EVAL] {msg}", flush=True)

# =========================
# Read videos
# =========================
VIDEO_PATH = ["metas/faceImg.mp4", "metas/leftImg.mp4", "metas/rightImg.mp4"]
p("Reading videos...")
video_data = []
for vp in VIDEO_PATH:
    t0 = time.time()
    frames = read_video_to_frames(vp)
    p(f"Loaded {vp}: T={frames.shape[0]}, frame0={frames[0].shape} {frames[0].dtype}, dt={time.time()-t0:.2f}s")
    video_data.append(frames)

T = min(v.shape[0] for v in video_data)
INSTRUCTION = "Connect two Lego pieces together"
p(f"Aligned T={T}")

# =========================
# Read gt actions
# =========================
data_file = "metas/actions.json"
Action_key = ["follow_left_joint_pos","follow_right_joint_pos","follow_left_gripper","follow_right_gripper"]

p(f"Reading actions: {data_file}")
gt_data = json.load(open(data_file, "r"))["data"]
p(f"gt frames={len(gt_data)}")

abs_traj = {k: [] for k in Action_key}
for frame_data in gt_data:
    for k in Action_key:
        v = frame_data[k]
        if not isinstance(v, list): v = [v]
        abs_traj[k].append(np.array(v))

abs_traj = {k: np.stack(v) for k, v in abs_traj.items()}
abs_traj = np.concatenate([abs_traj[k] for k in Action_key], axis=-1)

T = min(T, abs_traj.shape[0])
abs_traj = abs_traj[:T]
p(f"abs_traj: shape={abs_traj.shape}, dtype={abs_traj.dtype}")

# =========================
# WS client
# =========================
HOST, PORT = "127.0.0.1", 8010
p(f"Init HTTP client -> http://{HOST}:{PORT}/act")
pred_action_buffer = ActionBuffer(merge_strategy="replace")
client = HttpClient(action_buffer=pred_action_buffer, host=HOST, port=PORT)
p("Client ready")

# =========================
# Main loop
# =========================
for i in range(0, T, 30):
    payload = {"language_instruction": INSTRUCTION, "proprio": abs_traj[i], "domain": 0}
    for vid in range(len(video_data)):
        payload[f"image{vid}"] = video_data[vid][i]

    # p(f"step={i} proprio={payload['proprio'].shape} img0={payload['image0'].shape} {payload['image0'].dtype}")
    
    t0 = time.time()
    client.update(payload)
    for _ in range(30):
        client.get_action()
    p(f"step={i} update ok dt={(time.time()-t0)*1000:.1f}ms")

# =========================
# Snapshot + plot
# =========================
p("Snapshot buffer...")
pred_actions = pred_action_buffer.snapshot()
p(f"pred_actions: shape={pred_actions.shape}, dtype={pred_actions.dtype}")

import matplotlib.pyplot as plt
os.makedirs("exp", exist_ok=True)
L = min(pred_actions.shape[0], abs_traj.shape[0])
ts = np.arange(L)



fig, axs = plt.subplots(pred_actions.shape[1], 1, figsize=(10, 2*pred_actions.shape[1]), sharex=True)
if pred_actions.shape[1] == 1: axs = [axs]
for d in range(16):
    axs[d].plot(ts, pred_actions[:L, d], label="pred")
    axs[d].plot(ts, abs_traj[:L, d], label="gt", alpha=0.5)
    axs[d].legend()

plt.tight_layout()
out = "exp/predicted_vs_groundtruth_actions.png"
plt.savefig(out, dpi=200)
p(f"Saved: {out}")
