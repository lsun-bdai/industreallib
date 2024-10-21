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
from rclpy.executors import SingleThreadedExecutor
import numpy as np
from scipy.spatial.transform import Rotation
from threading import Thread
from franka_msgs.msg import GraspEpsilon

class FrankaArm:
    def __init__(self, 
                 node_name: str = "franka_arm",
                 init_ros: bool = True,
                 use_gripper: bool = False,
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
        self.gain_frequency = 1
        self.cartesian_stiffness = np.array([1024.0, 1024.0, 1024.0, 49.0, 49.0, 49.0])
        self.cartesian_damping = np.array([64.0, 64.0, 64.0, 14.0, 14.0, 14.0])
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
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        executor.add_node(self._state_client.node)
        thread = Thread(target=spin_node, args=(executor,), daemon=True)
        thread.start()
    
    def end_executor(self):
        self.node.destroy_node()
        self._state_client.node.destroy_node()
        rclpy.shutdown()

    
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
    
    def goto_pose(self, 
                  ee_pose,
                  ):
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

    def goto_delta_pose(self,
                        delta_ee_pose,
                        ):
        # get current end effector pose
        current_ee_pose = self._state_client.get_ee_pose()
        # pose is in position x, y, z, quaternion x, y, z, w
        # delta_ee_pose is in position x, y, z, euler x, y, z
        # get target position
        delta_ee_position = np.array(delta_ee_pose[0:3])
        target_ee_position = delta_ee_position + current_ee_pose[0:3]
        current_ee_orientation = Rotation.from_quat(current_ee_pose[3:7], scalar_first=True).as_quat() # xyzs
        delta_ee_orientation = np.array(delta_ee_pose[3:6])
        # get target orientation
        target_ee_orientation = (
            Rotation.from_euler("xyz", delta_ee_orientation) *
            Rotation.from_quat(current_ee_orientation, scalar_first=False)
        ).as_quat(scalar_first=False)  # xyzs
        target_ee_pose = np.concatenate([target_ee_position, target_ee_orientation])
        self.goto_pose(target_ee_pose)

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
    
    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        self.stop_cartesian_impedance()
        trajectory = {}
        reset_joint_target = [0.0, 0.0, 0.0, -2.34, 0.0, 2.30, 0.77]
        trajectory["position"] = np.array([reset_joint_target])
        trajectory["velocity"] = np.zeros((1, len(reset_joint_target)))
        trajectory["times"] = np.array([[3.0]])
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
    arm = FrankaArm(use_gripper=False)
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

