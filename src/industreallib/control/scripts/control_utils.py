"""IndustRealLib: Control utilities module.

This module defines utility functions for controlling a Franka robot with
the FrankaArm class.
"""

# Standard Library
import random
import signal

# Third Party
import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from bdai_msgs.msg import CartesianImpedanceGoal, CartesianImpedanceGain
from scipy.spatial.transform import Rotation


def open_gripper(franka_arm):
    """Opens the gripper."""
    print("\nOpening gripper...")
    franka_arm.open_gripper()
    print("Opened gripper.")


def close_gripper(franka_arm):
    """Closes the gripper."""
    print("\nClosing gripper...")
    franka_arm.close_gripper()
    print("Closed gripper.")


def go_to_joint_angles(franka_arm, joint_angles, duration):
    """Goes to a specified set of joint angles."""
    print("\nGoing to goal joint angles...")
    franka_arm.goto_joints(joint_angles, duration=duration)
    print("Finished going to goal joint angles.")

    print_joint_angles(franka_arm=franka_arm)


def go_to_pos(franka_arm, pos, duration):
    """Goes to a specified position, with gripper pointing downward."""
    # Compose goal pose
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = pos
    q = Rotation.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q

    print("\nGoing to goal position...")
    franka_arm.goto_pose(pose, duration=duration)
    print("Finished going to goal position.")

    curr_pose = franka_arm.get_ee_pose()
    print("\nCurrent position:", curr_pose[:3])


def go_to_pose(franka_arm, pos, ori_mat, duration, use_impedance):
    """Goes to a specified pose."""
    # Compose goal pose
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = pos
    q = Rotation.from_matrix(ori_mat).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q

    print("\nGoing to goal pose...")
    franka_arm.goto_pose(pose, duration=duration)
    print("Finished going to goal pose.")

    print_pose(franka_arm=franka_arm)


def go_home(franka_arm, duration):
    """Goes to a hard-coded home configuration."""
    print("\nGoing to home configuration...")
    go_to_joint_angles(
        franka_arm=franka_arm,
        joint_angles=[0.0, -1.76076077e-01, 0.0, -1.86691416e00, 0.0, 1.69344379e00, np.pi / 4],
        duration=duration,
    )
    print("Reached home configuration.")


def go_upward(franka_arm, dist, duration):
    """Goes upward by a specified distance while maintaining gripper orientation."""
    print("\nGoing upward...")
    franka_arm.goto_delta_pose([0.0, 0.0, dist, 0.0, 0.0, 0.0])
    print("Finished going upward.")


def go_downward(franka_arm, dist, duration):
    """Goes downward by a specified distance while maintaining gripper orientation."""
    print("\nGoing downward...")
    franka_arm.goto_delta_pose([0.0, 0.0, -dist, 0.0, 0.0, 0.0])
    print("Finished going downward.")


def get_pose_from_guide_mode(franka_arm, max_duration):
    """Activates guide mode. When complete, gets the gripper pose."""
    print("\nStarted guide mode.")

    input("Caution: Robot may drop. Press Enter to continue...")
    # Implement guide mode functionality here
    franka_arm.start_guide_mode()

    input(f"Robot is in guide mode for max duration of {max_duration} seconds. Press Enter to terminate...")
    franka_arm.stop_guide_mode()

    print("Stopped guide mode.")

    curr_pose = franka_arm.get_ee_pose()
    print_pose(franka_arm=franka_arm)

    return curr_pose


def print_joint_angles(franka_arm):
    """Prints the current joint angles."""
    curr_ang = franka_arm.get_joint_positions()
    print("\nCurrent joint angles:\n", curr_ang)


def print_pose(franka_arm):
    """Prints the current end-effector pose."""
    curr_pose = franka_arm.get_ee_pose()
    print("\nCurrent pose:")
    print("-------------")
    print("Position:", curr_pose[:3])
    print("Orientation (quaternion):", curr_pose[3:])
    print("Orientation (matrix):\n", Rotation.from_quat(curr_pose[3:]).as_matrix())


def get_pose_error(curr_pos, curr_ori_mat, targ_pos, targ_ori_mat):
    """Gets the error between a current pose and a target pose."""
    # Compute position error
    pos_err = np.linalg.norm(targ_pos - curr_pos)

    # Compute orientation error in radians
    ori_err_rad = (
        Rotation.from_matrix(targ_ori_mat) * Rotation.from_matrix(curr_ori_mat).inv()
    ).magnitude()

    return pos_err, ori_err_rad


def print_pose_error(curr_pos, curr_ori_mat, targ_pos, targ_ori_mat):
    """Prints the current pose, the target pose, and the error between the two poses."""
    print("\nCurrent pose:")
    print("-------------")
    print("Position:", curr_pos)
    print("Orientation:\n", curr_ori_mat)

    print("\nTarget pose:")
    print("-------------")
    print("Position:", targ_pos)
    print("Orientation:\n", targ_ori_mat)

    pos_err, ori_err_rad = get_pose_error(
        curr_pos=curr_pos, curr_ori_mat=curr_ori_mat, targ_pos=targ_pos, targ_ori_mat=targ_ori_mat
    )
    print("\nPose error:")
    print("-------------")
    print("Position:", pos_err)
    print("Orientation:", ori_err_rad)


def perturb_xy_pos(franka_arm, radial_bound):
    """Randomly perturbs the xy-position within a specified radius."""
    # Use rejection sampling to randomly sample delta_x and delta_y within circle
    curr_dist = np.inf
    while curr_dist > radial_bound:
        # Sample delta_x and delta_y within bounding square
        delta_x = random.uniform(-radial_bound, radial_bound)
        delta_y = random.uniform(-radial_bound, radial_bound)
        curr_dist = np.linalg.norm([delta_x, delta_y])

    print("\nPerturbing xy-position...")
    franka_arm.goto_delta_pose([delta_x, delta_y, 0.0, 0.0, 0.0, 0.0])
    print("Finished perturbing xy-position.")


def perturb_z_pos(franka_arm, bounds):
    """Randomly perturbs the z-position within a specified range."""
    delta_z = random.uniform(bounds[0], bounds[1])

    print("\nPerturbing z-position...")
    franka_arm.goto_delta_pose([0.0, 0.0, delta_z, 0.0, 0.0, 0.0])
    print("Finished perturbing z-position.")


def perturb_yaw(franka_arm, bounds):
    """Randomly perturbs the gripper yaw angle within a specified range."""
    delta_yaw = random.uniform(bounds[0], bounds[1])

    print("\nPerturbing yaw...")
    franka_arm.goto_delta_pose([0.0, 0.0, 0.0, 0.0, 0.0, delta_yaw])
    print("Finished perturbing yaw.")


def get_vec_rot_mat(unit_vec_a, unit_vec_b):
    """Gets rotation matrices that rotate one set of unit vectors to another set of unit vectors."""
    # Compute rotation axes (axis = u cross v / norm(u cross v))
    cross_prod = np.cross(unit_vec_a, unit_vec_b)  # (num_vecs, 3)
    cross_prod_norm = np.expand_dims(np.linalg.norm(cross_prod, axis=1), axis=1)  # (num_vecs, 1)
    rot_axis = cross_prod / cross_prod_norm  # (num_vecs, 3)

    # Compute rotation angles (theta = arccos(u dot v))
    rot_angle = np.expand_dims(
        np.arccos(np.einsum("ij,ij->i", unit_vec_a, unit_vec_b)), axis=1
    )  # (num_vecs, 1)

    # Compute axis-angle representation of rotation
    rot_axis_angle = rot_axis * rot_angle  # (num_vecs, 3)

    # Compute rotation matrix
    rot_mat = Rotation.from_rotvec(rot_axis_angle).as_matrix()  # (num_vecs, 3, 3)

    return rot_mat


def compose_cartesian_impedance_msg(targ_pos, targ_ori_quat, stiffness, damping):
    """Composes a CartesianImpedanceGoal message for task-space impedance control."""
    goal_msg = CartesianImpedanceGoal()
    goal_msg.pose.position.x, goal_msg.pose.position.y, goal_msg.pose.position.z = targ_pos
    goal_msg.pose.orientation.x, goal_msg.pose.orientation.y, goal_msg.pose.orientation.z, goal_msg.pose.orientation.w = targ_ori_quat

    gain_msg = CartesianImpedanceGain()
    gain_msg.stiffness = stiffness
    gain_msg.damping = damping

    return goal_msg, gain_msg


def set_sigint_response(franka_arm):
    """Sets a custom response to a SIGINT signal, which is executed on Ctrl + C."""

    def handler(signum, frame):
        """Defines a custom handler that stops the FrankaArm skill."""
        franka_arm.stop_skill()
        raise KeyboardInterrupt  # default behavior

    signal.signal(signal.SIGINT, handler)
