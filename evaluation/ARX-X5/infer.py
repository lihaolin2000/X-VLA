from robocontrol.xrrobot import RobotAccessor






def main(args):

    robot_controller = RobotAccessor().get_robot("desktop")
    if robot_controller is None:
        raise RuntimeError("Failed to get robot 'desktop'")
    
    robot_controller.left_arm.set_mode(args.control_mode)
    robot_controller.right_arm.set_mode(args.control_mode)






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start EX001 Dual Arm Robot Client")

    parser.add_argument(
        "--control-mode",
        type=str,
        default="end_pose",
        choices=["end_pose", "joints_pos"],
        help="Control mode: end_pose or joints",    
    )

    args = parser.parse_args()

    main(args)
