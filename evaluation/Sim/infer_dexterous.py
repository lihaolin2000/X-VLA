# Import necessary modules
from evaluation.websocket_client import WebSocketClient, HttpClient, ActionBuffer
import json_numpy
import numpy as np
from genmanip_client import EvalClient

# Initialize the evaluation client
eval_client = EvalClient(
                    base_url="http://123.57.187.96:55001",  # The base URL for the evaluation server
                    worker_ids=["0"],  # Worker IDs
                    config="task1.yml"  # Load a new task configuration file
                )

# Action buffer to handle action merging strategy
action_buffer = ActionBuffer(merge_strategy="replace")

# Initialize the model client
model_client = WebSocketClient(
    action_buffer,
    "10.140.60.112",
    8010)

try:
    print("Resetting the environment...")
    obs = eval_client.reset()  # Reset environment and get initial observation
    print("Environment reset complete. Starting action loop.")

    while True:
        # Fetch the action from the model
        action = model_client.get_action()

        if action is None:
            print("No action received, sending data to the model.")
            # Extract relevant data from the current observation
            image = obs['0']['obs']['camera_data']
            instruction = obs['0']['obs']['instruction']
            proprio = np.asarray(obs['state.joints'])

            # Prepare the payload for the model
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
        base_motion = action[14:20]

        # Process gripper actions
        if left_gripper < 0.:
            left_gripper = [0.04, 0.04]  # Close gripper
        else:
            left_gripper = [0.0, 0.0]  # Open gripper
        
        if right_gripper < 0.:
            right_gripper = [0.04, 0.04]  # Close gripper
        else:
            right_gripper = [0.0, 0.0]  # Open gripper
        
        # Format the action for submission
        format_action = {
            'action': left_joint.tolist() + left_gripper + right_joint.tolist() + right_gripper,
            'base_motion': base_motion.tolist(),
            'control_type': "joint_position"
        }

        print("Submitting action to environment.")
        obs, done = eval_client.step(format_action)  # Submit action and get the next observation

        # Check if the task is done
        if done:
            print("Task completed.")
            break

finally:
    print("Cleaning up and killing workers...")
    eval_client.kill_workers()
    print("Client cleaned.")
