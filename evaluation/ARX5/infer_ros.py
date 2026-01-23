import rospy
import cv2
from arm_control.msg import JointControl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from evaluation.websocket_client import WebSocketClient, HttpClient, ActionBuffer
import numpy as np

class X5Ros():
    def __init__(self, nodeName='ArmSlaveNode'):
        rospy.init_node(nodeName, anonymous=True)
        self._left_joint_pub = rospy.Publisher('/joint_control', JointControl, queue_size=10)
        self._right_joint_pub = rospy.Publisher('/joint_control2', JointControl, queue_size=10)
        self._left_joint_sub = rospy.Subscriber('/joint_information',JointControl,self.left_joints_callback)
        self._right_joint_sub = rospy.Subscriber('/joint_information2',JointControl,self.right_joints_callback)
        self._left_joints_state = []
        self._right_joints_state = []
        
        self.bridge = CvBridge()
        self.left_image = rospy.Subscriber('/camera1/usb_cam1/image_raw',Image,self.left_image_callback)
        self.center_image = rospy.Subscriber('/camera2/usb_cam2/image_raw',Image,self.center_image_callback)
        self.right_image = rospy.Subscriber('/camera3/usb_cam3/image_raw',Image,self.right_image_callback)
        
    def left_joints_callback(self,data):
        self._left_joints_state = data.joint_pos

    def right_joints_callback(self,data):
        self._right_joints_state = data.joint_pos

    def left_image_callback(self,data):
        self.left_camera_image = self.bridge.imgmsg_to_cv2(data.data, "bgr8")
        self.left_camera_image = cv2.resize(self.left_camera_image, (256, 256))
        self.left_camera_image = cv2.cvtColor(self.left_camera_image, cv2.COLOR_BGR2RGB)

    def center_image_callback(self,data):
        self.head_camera_image = self.bridge.imgmsg_to_cv2(data.data, "bgr8")
        self.head_camera_image = cv2.resize(self.head_camera_image, (256, 256))
        self.head_camera_image = cv2.cvtColor(self.head_camera_image, cv2.COLOR_BGR2RGB)

    def right_image_callback(self,data):
        self.right_camera_image = self.bridge.imgmsg_to_cv2(data.data, "bgr8")
        self.right_camera_image = cv2.resize(self.right_camera_image, (256, 256))
        self.right_camera_image = cv2.cvtColor(self.right_camera_image, cv2.COLOR_BGR2RGB)


    def send_left_joint_command(self, joint_positions):
        msg = JointControl()
        msg.joint_pos = joint_positions
        msg.joint_vel = [0.0] * 7
        msg.joint_cur = [0.0] * 7
        self._left_joint_pub.publish(msg)

    def send_right_joint_command(self, joint_positions):
        msg = JointControl()
        msg.joint_pos = joint_positions
        msg.joint_vel = [0.0] * 7
        msg.joint_cur = [0.0] * 7
        self._right_joint_pub.publish(msg)

def get_observation(robot_controller):
    observation = {
        "left_joints_pos": robot_controller._left_joints_state,
        "right_joints_pos": robot_controller._right_joints_state,
        "left_camera_image": robot_controller.left_camera_image,
        "right_camera_image": robot_controller.right_camera_image,
        "head_camera_image": robot_controller.head_camera_image,
    }
    return observation

def main(args):
    assert args.control_mode == "joints_pos", \
        "Only joints_pos control mode is supported in this script."

    robot_controller = X5Ros()
    if robot_controller is None:
        raise RuntimeError("Failed to get robot 'desktop'")
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
                observation["right_joints_pos"]
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
            robot_controller.send_left_joint_command(action[:7])
            robot_controller.send_right_joint_command(action[7:14])

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
                        default="http",
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
