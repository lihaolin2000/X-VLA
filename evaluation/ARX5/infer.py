from robocontrol.xrrobot import RobotAccessor
from evaluation.websocket_client import WebSocketClient, HttpClient, ActionBuffer
import numpy as np
def get_observation(robot_controller):
    l_ee_pose = robot_controller.left_arm.get_state_end_pose()
    r_ee_pose = robot_controller.right_arm.get_state_end_pose()
    l_joints_pos = robot_controller.left_arm.get_state_joints_pos()
    r_joints_pos = robot_controller.right_arm.get_state_joints_pos()
    l_gripper = robot_controller.left_ee.get_state()
    r_gripper = robot_controller.right_ee.get_state()
    
    left_image = robot_controller.left_camera.get_raw_image()
    right_image = robot_controller.right_camera.get_raw_image()
    head_image = robot_controller.head_camera.get_raw_image()
    observation = {
        "left_ee_pose": l_ee_pose,
        "right_ee_pose": r_ee_pose,
        "left_joints_pos": l_joints_pos,
        "right_joints_pos": r_joints_pos,
        "left_gripper": l_gripper,
        "right_gripper": r_gripper,
        "left_camera_image": left_image,
        "right_camera_image": right_image,
        "head_camera_image": head_image,
    }
    return observation

def main(args):

    assert args.control_mode == "joints_pos", \
        "Only joints_pos control mode is supported in this script."

    robot_controller = RobotAccessor().get_robot("desktop")
    if robot_controller is None:
        raise RuntimeError("Failed to get robot 'desktop'")

    robot_controller.left_arm.set_mode(args.control_mode)
    robot_controller.right_arm.set_mode(args.control_mode)
    print("Robot controller initialized with control mode:", args.control_mode)

    #=====================================================================
    #===  init model client                                           ===
    #=====================================================================
    action_buffer = ActionBuffer(merge_strategy="replace")
    if args.client_type == "ws":
        model_client = WebSocketClient(
            action_buffer=action_buffer,
            host=args.model_address,
            port=args.port
        )
    else:  # http client
        model_client = HttpClient(
            action_buffer=action_buffer,
            server_address=args.model_address,
            server_port=args.port
        )   
    print(f"Model client ({args.client_type}) initialized and connected to {args.model_address}:{args.port}")
    while True:
        observation = get_observation(robot_controller)
        payload = {
            "image0": observation["head_camera_image"],
            "image1": observation["left_camera_image"],
            "image2": observation["right_camera_image"],


            "proprio": np.concatenate([
                observation["left_joints_pos"],
                observation["right_joints_pos"],
                [observation["left_gripper"]],
                [observation["right_gripper"]],
            ]),
            "language_instruction": args.language_instruction,
            "domain": 0,
        }
        model_client.update(payload, sync=True)
        while True:
            action = model_client.get_action()
            if action is not None: break
            print("get action:", action)
            if args.debug_mode:      
                print("current proprio:", payload["proprio"])          
                input("Debug mode: Press Enter to send action...")
            robot_controller.left_arm.send_cmd_joints_pos(action[:7])
            robot_controller.right_arm.send_cmd_joints_pos(action[7:14])
            robot_controller.left_ee.send_cmd(action[14])
            robot_controller.right_ee.send_cmd(action[15])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start EX001 Dual Arm Robot Client")

    parser.add_argument(
        "--control-mode",
        type=str,
        default="joints_pos",
        choices=["end_pose", "joints_pos"],
        help="Control mode: end_pose or joints",    
    )
    parser.add_argument("client_type", 
                        type=str, 
                        choices=["http", "ws"], 
                        help="Type of client to use: http or ws"
    )
    parser.add_argument("--language-instruction", 
                        type=str, 
                        default="Connect two Lego pieces together", 
                        help="Language instruction for the model"
    )
    parser.add_argument("--model-address", 
                        default="localhost", 
                        type=str, 
                        help="Model server IP address"
    )
    parser.add_argument("--port", 
                        type=int, 
                        default=8000, 
                        help="Model server port"
    )
    parser.add_argument("--debug_mode", 
                        action="store_true", 
                        help="Enable debug mode")
    args = parser.parse_args()
    main(args)
