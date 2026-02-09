# Import necessary modules
from evaluation.websocket_client import WebSocketClient, HttpClient, ActionBuffer
import json_numpy
import msgpack
import msgpack_numpy as m
m.patch()
import numpy as np
from genmanip_client import EvalClient
import requests
import av
import numpy as np
# Initialize the evaluation client
worker_ids=["1"]
eval_client = EvalClient(
                    base_url="http://dsw-notebook-dsw-fh4w05j2tmvgkh5xtt-55001.vpc-2zef1skt5zeyxqsntfobm.instance-forward.dsw.cn-beijing.aliyuncs.com:55001",  # The base URL for the evaluation server
                    worker_ids=worker_ids,  # Worker IDs
                    config="configs/tasks/ebench/long/bottle.yml"  # Load a new task configuration file
                )

# Initialize the model client
action_buffer = ActionBuffer()
model_client = HttpClient(
    action_buffer,
    "127.0.0.1",
    8010)


saved_video = []
try:
    print("Resetting the environment...")
    obs = eval_client.reset()  # Reset environment and get initial observation
    print("Environment reset complete. Starting action loop.")
    
    while True:
        # Fetch the action from the model

        if action_buffer.left_valid_time() < 1:
            print("No action received, sending data to the model.")
            # Extract relevant data from the current observation
            print(obs[worker_ids[0]]['obs'].keys())
            image = obs[worker_ids[0]]['obs']['camera_data']
            instruction = obs[worker_ids[0]]['obs']['instruction']
            proprio = np.asarray(obs[worker_ids[0]]['obs']['state.joints'])
            with open("log.txt", "a+") as f:
                proprio = obs[worker_ids[0]]['obs']['state.joints'].tolist()
                f.write(f"{len(proprio)}\n")
            payload = {
                "proprio": json_numpy.dumps(proprio),  # Joint state
                "language_instruction": instruction,  # Instruction
                "image0": json_numpy.dumps(image['top_camera']['rgb']),
                "image1": json_numpy.dumps(image['left_camera']['rgb']),
                "image2": json_numpy.dumps(image['right_camera']['rgb']),
                "domain_id": 0,
                "steps": 10
            }
            
            # Send data to the model and get the next action
            model_client.update(payload)
        action = model_client.get_action()

        # Process the action
        left_joint = action[:6]
        right_joint = action[6:12]
        left_gripper = action[12]
        right_gripper = action[13]
        base_motion = action[14:17]

        # Process gripper actions
        if left_gripper < -0.9:
            left_gripper = [0.05, 0.05]  # open gripper
        else:
            left_gripper = [-0.01, -0.01]  # close gripper
        
        if right_gripper < -0.9:
            right_gripper = [0.05, 0.05]  # open gripper
        else:
            right_gripper = [-0.01, -0.01]  # close gripper
        


        # Format the action for submission
        format_action = {worker_ids[0]:{
            'action': left_joint.tolist() + left_gripper + right_joint.tolist() + right_gripper,
            'base_motion': base_motion.tolist(),
            'control_type': "joint_position"
        }}
        
        print("Submitting action to environment.")
        obs, done = eval_client.step(format_action)  # Submit action and get the next observation

        action_slice, start_idx = action_buffer.snapshot()

        data_to_send = {
            "image_top": obs[worker_ids[0]]['obs']['camera_data']['top_camera']['rgb'],
            "image_left": obs[worker_ids[0]]['obs']['camera_data']['left_camera']['rgb'],
            "image_right": obs[worker_ids[0]]['obs']['camera_data']['right_camera']['rgb'],
            "telemetry": action_slice,
            "start_idx": start_idx,
            "step_idx": action_buffer.current_time             
        }
        response = requests.post("http://127.0.0.1:8080/upload",
                            data=msgpack.packb(data_to_send, use_bin_type=True))
        # Check if the task is done
        # if obs[worker_ids[0]]['obs']['reset']:
        #     action_buffer.reset()
        if done:
            print("Task completed.")
            break
finally:

    print("Cleaning up and killing workers...")
    eval_client.kill_workers()
    print("Client cleaned.")
