"""
ROS 2 node that bridge franka_arm and policy level actions.

Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
"""

import argparse
import os
import sys
import time
from rclpy.node import Node
from pathlib import Path
from typing import Any
import rclpy
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from industreallib.robot.franka_arm_state_client import FrankaArmStateClient
from rclpy.action import ActionClient
from controller_manager_msgs.srv import SwitchController
from franka_msgs.msg import FrankaRobotState
from bdai_msgs.msg import CartesianImpedanceGoal, CartesianImpedanceGain
from typing import Optional
from franka_msgs.action import Grasp, Homing, Move
from time import sleep
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from threading import Thread
from franka_msgs.msg import GraspEpsilon
from industreallib.robot.franka_arm_state_client import FrankaConstants
class FrankaArm:
    def __init__(self, 
                 node_name: str = "franka_arm",
                 init_ros: bool = True,
                 use_gripper: bool = True,
                 reset_robot_on_init: bool = True,
                 ):
        """
        Initialize a FrankaArm.
        """
        # init ROS
        if init_ros:
            rclpy.init(args=None)
        self.use_gripper = use_gripper
        self.node = Node(node_name)
        
        # Franka Arm State Client to get joint states, gripper states, and robot state
        self._state_client = FrankaArmStateClient()
        
        # FrankaArm Joint trajectory control
        self.arm_trajectory_cli = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/franka_joint_trajectory_controller/follow_joint_trajectory",            
        )
        # FrankaArm Cartesian impedance control
        self.cartesian_impedance_goal: Optional[CartesianImpedanceGoal] = None
        self.cartesian_impedance_goal_publisher = self.node.create_publisher(
            CartesianImpedanceGoal,
            "/franka_cartesian_impedance_controller/commands",
            1000,
        )
        self.cartesian_impedance_gain_publisher = self.node.create_publisher(
            CartesianImpedanceGain, 
            "/franka_cartesian_impedance_controller/gains", 
            10
        )

        # Arm Parameters for Control of End Effector
        self.gain_frequency = 1
        self.control_frequency = 50
        self.cartesian_stiffness = np.array([1024.0, 1024.0, 1024.0, 49.0, 49.0, 49.0])
        self.cartesian_damping = np.array([64.0, 64.0, 64.0, 14.0, 14.0, 14.0])
        self.max_linear_speed = 0.2  # m/s
        self.max_angular_speed = 0.5  # rad/s
        
        # Publish gains at a fixed frequency
        self.cartesian_impedance_gain_publish_timer = self.node.create_timer(
            1/self.gain_frequency, 
            self.publish_gains,
        )

        # Switch controller client
        self.switch_cli = self.node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        # Wait for switch controller service to be available
        while not self.switch_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "switch controller service not available, waiting again..."
            )
        if self.use_gripper:
            self.gripper_move_action_client = ActionClientWrapper(
                Move, "/fr3_gripper/move", self.node
            )
            self.gripper_grasp_action_client = ActionClientWrapper(
                Grasp, "/fr3_gripper/grasp", self.node
            )
            self.gripper_homing_action_client = ActionClientWrapper(
                Homing, "/fr3_gripper/homing", self.node
            )
        self.start_executor()
        # run something to init the robot
        if reset_robot_on_init:
            self.reset_joint()
            if self.use_gripper:
                self.open_gripper()
            self.goto_delta_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            self.start_cartesian_impedance()
            self.goto_delta_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.guide_mode_timer = None

    def publish_gains(self):
        gain_msg = CartesianImpedanceGain()
        gain_msg.stiffness = self.cartesian_stiffness
        gain_msg.damping = self.cartesian_damping
        self.cartesian_impedance_gain_publisher.publish(gain_msg)

    def start_executor(self):
        def spin_node(executor):
            executor.spin()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self._state_client.node)
        try:    
            self.executor_thread = Thread(target=spin_node, args=(self.executor,), daemon=True)
            self.executor_thread.start()
        except KeyboardInterrupt:
            self.end_executor()
    
    def end_executor(self):
        # stop the thread and executor properly
        self.executor.shutdown()
        self.node.destroy_node()
        self._state_client.node.destroy_node()
        rclpy.shutdown()
        self.executor_thread.join()

    def create_rate(self, frequency=None):
        if frequency is None:
            frequency = self.control_frequency
        else:
            self.control_frequency = frequency

        self.control_rate = self.node.create_rate(frequency)
        return self.control_rate
    
    def get_logger(self):
        return self.node.get_logger()
    
    def get_time(self):
        return self.node.get_clock().now().nanoseconds / 1e9
    
    def start_cartesian_impedance(self):
        """Switch from joint trajectory controller to cartesian impedance controller"""
        print("Switching to cartesian impedance controller")
        switch_req = SwitchController.Request()
        switch_req.deactivate_controllers = ["franka_joint_trajectory_controller"]
        switch_req.activate_controllers = ["franka_cartesian_impedance_controller"]
        switch_req.strictness = SwitchController.Request.BEST_EFFORT
        switch_req.timeout = rclpy.duration.Duration(seconds=8).to_msg()
        switch_resp = self.switch_cli.call(switch_req)
        return switch_resp

    def stop_cartesian_impedance(self):
        """Switch from cartesian impedance controller to joint trajectory controller"""
        print("Switching to joint trajectory controller")
        switch_req = SwitchController.Request()
        switch_req.deactivate_controllers = ["franka_cartesian_impedance_controller"]
        switch_req.activate_controllers = ["franka_joint_trajectory_controller"]
        switch_req.strictness = SwitchController.Request.BEST_EFFORT
        switch_req.timeout = rclpy.duration.Duration(seconds=8).to_msg()
        switch_resp = self.switch_cli.call(switch_req)
        return switch_resp

    def adjust_cartesian_impedance(self,
                                   stiffness: np.ndarray,
                                   damping: np.ndarray,
                                   ):
        self.cartesian_stiffness = stiffness
        self.cartesian_damping = damping
       
    def is_skill_done(self):
        pass

    def stop_skill(self):
        """Stops the current skill and maintains the current position."""
        print("Stopping current skill and maintaining position")
        self.goto_delta_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def wait_for_skill(self):
        while not self.is_skill_done():
            continue
    
    def wait_for_gripper(self):
        if self._last_gripper_command == "Grasp":
            done = self._gripper_grasp_action_client.wait_for_result()
        elif self._last_gripper_command == "Homing":
            done = self._gripper_homing_action_client.wait_for_result()
        elif self._last_gripper_command == "Stop":
            done = self._gripper_stop_action_client.wait_for_result()
        elif self._last_gripper_command == "Move":
            done = self._gripper_move_action_client.wait_for_result()
        sleep(2)
    
    def goto_pose(self, ee_pose):
        """Goes to specified end-effector pose directly.
        
        Args:
            ee_pose: Target pose in format [x, y, z, qx, qy, qz, qw]
        """
        # Publish target pose
        self.cartesian_impedance_goal = CartesianImpedanceGoal()
        goal_pose = Pose()
        goal_pose.position.x = ee_pose[0]
        goal_pose.position.y = ee_pose[1]
        goal_pose.position.z = ee_pose[2]
        goal_pose.orientation.x = ee_pose[3]
        goal_pose.orientation.y = ee_pose[4]
        goal_pose.orientation.z = ee_pose[5]
        goal_pose.orientation.w = ee_pose[6]
        
        self.cartesian_impedance_goal.pose = goal_pose
        self.cartesian_impedance_goal_publisher.publish(self.cartesian_impedance_goal)

    def goto_pose_with_duration(self, ee_pose, duration=None):
        """Goes to specified end-effector pose with interpolation over given duration.
        
        Args:
            ee_pose: Target pose in format [x, y, z, qx, qy, qz, qw]
            duration: Time to reach target pose in seconds
        """
        # Get current pose
        current_ee_pose = self._state_client.get_ee_pose()

        # If no duration specified, check distance and set reasonable duration
        if duration is None:
            pos_dist = np.linalg.norm(ee_pose[:3] - current_ee_pose[:3])
            ori_dist = np.arccos(2 * np.power(np.dot(current_ee_pose[3:], ee_pose[3:]), 2) - 1)
            duration = max(pos_dist / self.max_linear_speed, ori_dist / self.max_angular_speed)

        # Calculate number of interpolation steps based on control frequency
        num_steps = int(duration * self.control_frequency)
        
        # Spherical interpolation for orientation
        current_rot = Rotation.from_quat(current_ee_pose[3:])
        target_rot = Rotation.from_quat(ee_pose[3:])
        slerp = Slerp([0, 1], Rotation.concatenate([current_rot, target_rot]))
        for step in range(num_steps + 1):
            # Linear interpolation for position
            alpha = step / num_steps
            interp_pos = current_ee_pose[:3] + alpha * (ee_pose[:3] - current_ee_pose[:3])
            interp_quat = slerp(alpha).as_quat()
            
            # Publish interpolated pose
            self.cartesian_impedance_goal = CartesianImpedanceGoal()
            goal_pose = Pose()
            goal_pose.position.x = interp_pos[0]
            goal_pose.position.y = interp_pos[1] 
            goal_pose.position.z = interp_pos[2]
            goal_pose.orientation.x = interp_quat[0]
            goal_pose.orientation.y = interp_quat[1]
            goal_pose.orientation.z = interp_quat[2]
            goal_pose.orientation.w = interp_quat[3]
            
            self.cartesian_impedance_goal.pose = goal_pose
            self.cartesian_impedance_goal_publisher.publish(self.cartesian_impedance_goal)
            
            self.control_rate.sleep()

    def goto_delta_pose(self,
                        delta_ee_pose,
                        duration=None,
                        ):
        # get current end effector pose
        current_ee_pose = self._state_client.get_ee_pose()
        # pose is in position x, y, z, quaternion x, y, z, w
        # delta_ee_pose is in position x, y, z, euler x, y, z
        # get target position
        delta_ee_position = np.array(delta_ee_pose[0:3])
        target_ee_position = delta_ee_position + current_ee_pose[0:3]
        current_ee_orientation = current_ee_pose[3:7] # xyzs
        delta_ee_orientation = np.array(delta_ee_pose[3:6])
        # get target orientation
        target_ee_orientation = (
            Rotation.from_euler("xyz", delta_ee_orientation) *
            Rotation.from_quat(current_ee_orientation)
        ).as_quat()  # xyzs
        target_ee_pose = np.concatenate([target_ee_position, target_ee_orientation])
        if duration is None:
            self.goto_pose(target_ee_pose)
        else:
            self.goto_pose_with_duration(target_ee_pose, duration)

    def goto_joints(self,
                    joint_trajectory,
                    ):
        goal_msg = FollowJointTrajectory.Goal()

        goal_msg.trajectory.joint_names = self._state_client.joint_names
        for i in range(joint_trajectory["times"].shape[0]):
            traj_point = JointTrajectoryPoint()
            traj_point.positions = []
            traj_point.velocities = []
            for j in range(len(goal_msg.trajectory.joint_names)):
                if "position" not in joint_trajectory and "velocity" not in joint_trajectory:
                    raise RuntimeError(
                        "trajectory dictionary must contain position or velocity"
                    )
                if "position" in joint_trajectory:
                    traj_point.positions.append(joint_trajectory["position"][i, j])
                if "velocity" in joint_trajectory:
                    traj_point.velocities.append(joint_trajectory["velocity"][i, j])
            traj_point.time_from_start = rclpy.duration.Duration(
                seconds=joint_trajectory["times"][i]
            ).to_msg()
            goal_msg.trajectory.points.append(traj_point)
        self.node.get_logger().info("sending trajectory")
        self.arm_trajectory_cli.send_goal(goal_msg)
    
    def reset_joint(self, duration=None):
        """Resets Joints (needed after running for hours)"""
        if duration is None:
            duration = 3.0
        self.stop_cartesian_impedance()
        trajectory = {}
        reset_joint_target = FrankaConstants.HOME_JOINTS_AMBER
        # reset_joint_target = [0.0, 0.0, 0.0, -2.34, 0.0, 2.30, 0.77]
        trajectory["position"] = np.array([reset_joint_target])
        trajectory["velocity"] = np.zeros((1, len(reset_joint_target)))
        trajectory["times"] = np.array([[duration]])
        self.goto_joints(trajectory)
        time.sleep(0.01)
        print("Reset trajectory complete")
        self.start_cartesian_impedance()
        print("Restarted cartesian controller")

    def goto_gripper(self, 
                     width, 
                     grasp=False, 
                     speed=0.04, 
                     force=0.0,
                     epsilon_inner=0.08,
                     epsilon_outer=0.08, 
                     block=True, 
                     ignore_errors=True, 
                     skill_desc='GoToGripper'):
        if grasp:
            goal_msg = Grasp.Goal(
                width=width, 
                speed=speed, 
                force=force, 
                epsilon=GraspEpsilon(inner=epsilon_inner, outer=epsilon_outer)
            )
            self.gripper_grasp_action_client.send_goal_and_wait(
                skill_desc, goal_msg, timeout_sec=5
            )
        else:
            goal_msg = Move.Goal(width=width, speed=speed)
            self.gripper_move_action_client.send_goal_and_wait(
                skill_desc, goal_msg, timeout_sec=5
            )

    def home_gripper(self, block=True, skill_desc='HomeGripper'):
        goal_msg = Homing.Goal()
        self.gripper_homing_action_client.send_goal_and_wait(
            skill_desc, goal_msg, timeout_sec=5
        )
    
    def open_gripper(self, block=True, skill_desc='OpenGripper'):
        self.goto_gripper(width=0.09, grasp=False, block=block, skill_desc=skill_desc)
    
    def close_gripper(self, grasp=True, block=True, skill_desc='CloseGripper'):
        self.goto_gripper(width=0.01, grasp=grasp, block=block, skill_desc=skill_desc)
    
    def start_guide_mode(self, skill_desc='GuideMode'):
        self.node.get_logger().info("Starting guide mode")
        self.node.get_logger().info("Press Enter to enter guide mode...")
        self.adjust_cartesian_impedance(stiffness=np.zeros(6), damping=np.zeros(6))        
        # Set a timer for maximum duration (60 seconds)
        if self.guide_mode_timer:
            self.guide_mode_timer.cancel()
        self.guide_mode_timer = self.node.create_timer(60.0, self.stop_guide_mode)
    
    def stop_guide_mode(self):
        if self.guide_mode_timer:
            self.guide_mode_timer.cancel()
            self.guide_mode_timer = None
        # set current pose as impedance controller goal to avoid drift
        self.goto_delta_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.adjust_cartesian_impedance(stiffness=np.array([1024.0, 1024.0, 1024.0, 49.0, 49.0, 49.0]), damping=np.array([64.0, 64.0, 64.0, 14.0, 14.0, 14.0]))
        self.publish_gains()
        self.node.get_logger().info("Guide mode terminated")
    
    def __del__(self): 
        self.end_executor()
    
    def __getattr__(self, name):
        if name.startswith('get_') and hasattr(self._state_client, name):
            return getattr(self._state_client, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

if __name__ == "__main__":
    arm = FrankaArm(use_gripper=True)
    arm.reset_joint()
    # move in turn along x, y, z and get back, then repeat at 5 Hz
    import time

    def move_with_delay(delta):
        arm.goto_delta_pose(delta)
        time.sleep(0.2)

    # Move one round along x, y, z axes
    move_with_delay([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    move_with_delay([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    move_with_delay([0.0, 0.0, 0.03, 0.0, 0.0, 0.0])
    move_with_delay([-0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    move_with_delay([0.0, -0.03, 0.0, 0.0, 0.0, 0.0])
    move_with_delay([0.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    print("Done")

