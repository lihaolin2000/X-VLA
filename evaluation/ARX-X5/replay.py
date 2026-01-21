from evaluation.utils import ActionBuffer, AutoVisualizer, WebSocketClient
from datasets.utils import read_video_to_frames
import numpy as np
import json
import time
# ======================================================================
# ===  Reading visual and lang observation                           ===
# ======================================================================
VIDEO_PATH = [
    "metas/video1.mp4",
    "metas/video2.mp4",
    "metas/video3.mp4",
]
video_data = [read_video_to_frames(p) for p in VIDEO_PATH]
INSTRUCTION = "Connect two Lego pieces together"
# ======================================================================
# ===  read groudtruth action file                                   ===
# ======================================================================
data_file = "metas/actions.json"
Action_key = ["follow_left_joint_pos", 
            "follow_right_joint_pos",
            "follow_left_gripper", 
            "follow_right_gripper"]
gt_data = json.load(open(data_file, "r"))
abs_traj = {key: [] for key in Action_key}
for frame_data in gt_data:
    for key in Action_key:
        if not isinstance(frame_data[key], list): frame_data[key] = [frame_data[key]]
        abs_traj[key].append(np.array(frame_data[key]))
abs_traj = {key: np.stack(abs_traj[key]) for key in Action_key}
abs_traj = np.concatenate([np.asarray(abs_traj[key]) for key in Action_key], axis=-1)
# ======================================================================
# ===  init websocket client
# ======================================================================
HOST = "127.0.0.1"
PORT = "8010"
pred_action_buffer = ActionBuffer(merge_strategy="replace")
gt_action_buffer = ActionBuffer(merge_strategy="replace")
client = WebSocketClient(action_buffer=pred_action_buffer,
                         host=HOST,
                         port=PORT)




















