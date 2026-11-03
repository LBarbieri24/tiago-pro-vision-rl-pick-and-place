#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class TIAGoProRLStart(Node):
    def __init__(self):
        super().__init__('tiago_pro_rl_start_node')
        
        self.get_logger().info("Initializing direct trajectory clients...")
        
        # Action Clients for different joint groups
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_right_controller/follow_joint_trajectory')
        self.arm_left_client = ActionClient(self, FollowJointTrajectory, '/arm_left_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_right_controller/follow_joint_trajectory')
        self.head_client = ActionClient(self, FollowJointTrajectory, '/head_controller/follow_joint_trajectory')
        self.torso_client = ActionClient(self, FollowJointTrajectory, '/torso_controller/follow_joint_trajectory')
        
    def execute(self):
        # 1. Wait for action servers
        self.get_logger().info("Waiting for joint controllers to be active...")
        controllers = [
            ("Arm Right", self.arm_client),
            ("Arm Left", self.arm_left_client),
            ("Gripper", self.gripper_client),
            ("Head", self.head_client),
            ("Torso", self.torso_client)
        ]
        for name, client in controllers:
            if not client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f"{name} controller action server NOT available!")
                return False
        
        # --- Arm Right Goal ---
        arm_goal = FollowJointTrajectory.Goal()
        arm_goal.trajectory.joint_names = [
            'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
            'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
            'arm_right_7_joint'
        ]
        arm_p = JointTrajectoryPoint()
        arm_p.positions = [-3.14, -1.7, 0.71, -1.14, 2.8, 1.0, 0.0]
        arm_p.time_from_start = Duration(sec=20, nanosec=0)
        arm_goal.trajectory.points = [arm_p]

        # --- Arm Left Goal ---
        arm_left_goal = FollowJointTrajectory.Goal()
        arm_left_goal.trajectory.joint_names = [
            'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
            'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint',
            'arm_left_7_joint'
        ]
        arm_left_p = JointTrajectoryPoint()
        arm_left_p.positions = [0.36, -1.83, 0.47, -2.35, 0.0, -1.20, 0.0]
        arm_left_p.time_from_start = Duration(sec=20, nanosec=0)
        arm_left_goal.trajectory.points = [arm_left_p]

        # --- Gripper Right Goal ---
        gripper_goal = FollowJointTrajectory.Goal()
        gripper_goal.trajectory.joint_names = ['gripper_right_finger_joint']
        grip_p = JointTrajectoryPoint()
        grip_p.positions = [0.045]  # Open gripper
        grip_p.time_from_start = Duration(sec=20, nanosec=0)
        gripper_goal.trajectory.points = [grip_p]

        # --- Head Goal ---
        head_goal = FollowJointTrajectory.Goal()
        head_goal.trajectory.joint_names = ['head_1_joint', 'head_2_joint']
        head_p = JointTrajectoryPoint()
        head_p.positions = [0.0, -0.785398]  # 45 deg tilt down
        head_p.time_from_start = Duration(sec=20, nanosec=0)
        head_goal.trajectory.points = [head_p]

        # --- Torso Goal ---
        torso_goal = FollowJointTrajectory.Goal()
        torso_goal.trajectory.joint_names = ['torso_lift_joint']
        torso_p = JointTrajectoryPoint()
        torso_p.positions = [0.0]  # Match MuJoCo training value
        torso_p.time_from_start = Duration(sec=20, nanosec=0)
        torso_goal.trajectory.points = [torso_p]

        # --- Send goals and spin until all goals are accepted and completed ---
        self.get_logger().info("Sending joint goals concurrently...")
        
        arm_future = self.arm_client.send_goal_async(arm_goal)
        arm_left_future = self.arm_left_client.send_goal_async(arm_left_goal)
        gripper_future = self.gripper_client.send_goal_async(gripper_goal)
        head_future = self.head_client.send_goal_async(head_goal)
        torso_future = self.torso_client.send_goal_async(torso_goal)
        
        # Wait for all goals to be accepted
        rclpy.spin_until_future_complete(self, arm_future)
        rclpy.spin_until_future_complete(self, arm_left_future)
        rclpy.spin_until_future_complete(self, gripper_future)
        rclpy.spin_until_future_complete(self, head_future)
        rclpy.spin_until_future_complete(self, torso_future)
        
        arm_handle = arm_future.result()
        arm_left_handle = arm_left_future.result()
        gripper_handle = gripper_future.result()
        head_handle = head_future.result()
        torso_handle = torso_future.result()
        
        if not (arm_handle.accepted and arm_left_handle.accepted and gripper_handle.accepted and head_handle.accepted and torso_handle.accepted):
            self.get_logger().error("One or more goals were REJECTED by controllers!")
            return False
            
        self.get_logger().info("Goals accepted! Moving to rl_start position (takes 8s)...")
        
        # Wait for trajectory execution to finish
        arm_result_future = arm_handle.get_result_async()
        arm_left_result_future = arm_left_handle.get_result_async()
        gripper_result_future = gripper_handle.get_result_async()
        head_result_future = head_handle.get_result_async()
        torso_result_future = torso_handle.get_result_async()
        
        rclpy.spin_until_future_complete(self, arm_result_future)
        rclpy.spin_until_future_complete(self, arm_left_result_future)
        rclpy.spin_until_future_complete(self, gripper_result_future)
        rclpy.spin_until_future_complete(self, head_result_future)
        rclpy.spin_until_future_complete(self, torso_result_future)
        
        self.get_logger().info("Successfully reached rl_start position!")
        return True

def main(args=None):
    rclpy.init(args=args)
    node = TIAGoProRLStart()
    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
