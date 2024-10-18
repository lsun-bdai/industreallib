import numpy as np
import rclpy
from industreallib.robot.franka_arm import FrankaArm

def cartesian_position_error_test():
    arm = FrankaArm(use_gripper=False)
    arm.reset_joint()
    # move in turn along x, y, z and get back, then repeat at 5 Hz
    import time

    loop_num = 0
    initial_pose = arm._state_client.get_ee_pose()
    while loop_num < 100:
        start_time = time.time()
        def move_with_delay(delta):
            start = time.time()
            arm.goto_delta_pose(delta)
            elapsed = time.time() - start
            if elapsed < 0.2:
                time.sleep(0.2 - elapsed)

        move_with_delay([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
        move_with_delay([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
        move_with_delay([0.0, 0.0, 0.03, 0.0, 0.0, 0.0])
        move_with_delay([-0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
        move_with_delay([0.0, -0.03, 0.0, 0.0, 0.0, 0.0])
        move_with_delay([0.0, 0.0, -0.03, 0.0, 0.0, 0.0])
        loop_num += 1
        if loop_num % 10 == 0:
            print(f"Loop {loop_num} complete")
            current_pose = arm._state_client.get_ee_pose()
            pose_error = np.linalg.norm(np.array(current_pose) - np.array(initial_pose))
            pose_diff = np.array(current_pose) - np.array(initial_pose)
            print(f"Pose error after {loop_num} loops:")
            print(f"  X: {pose_diff[0]:.6f}")
            print(f"  Y: {pose_diff[1]:.6f}")
            print(f"  Z: {pose_diff[2]:.6f}")
            print(f"  Rotation X: {pose_diff[3]:.6f}")
            print(f"  Rotation Y: {pose_diff[4]:.6f}")
            print(f"  Rotation Z: {pose_diff[5]:.6f}")
            print(f"  Total error: {pose_error:.6f}")
        if loop_num % 10 == 0:
            if input("Press 'q' to stop, or any other key to continue: ").lower() == 'q':
                break
    
    print("Cartesian Position Error Test Done")
    rclpy.shutdown()

def cartesian_orientation_error_test():
    arm = FrankaArm(use_gripper=False)
    arm.reset_joint()
    
    import time
    from scipy.spatial.transform import Rotation

    loop_num = 0
    initial_pose = arm._state_client.get_ee_pose()
    initial_rotation = Rotation.from_quat(initial_pose[3:])

    while loop_num < 100:
        def rotate_with_delay(rotation_delta_deg):
            start = time.time()
            rotation_delta_rad = np.deg2rad(rotation_delta_deg)
            delta_ee_pose = [0.0, 0.0, 0.0, rotation_delta_rad[0], rotation_delta_rad[1], rotation_delta_rad[2]]
            arm.goto_delta_pose(delta_ee_pose)
            elapsed = time.time() - start
            if elapsed < 0.2:
                time.sleep(0.2 - elapsed)

        rotate_with_delay([5, 0, 0])  # Rotate 5 degrees around x-axis
        rotate_with_delay([0, 5, 0])  # Rotate 5 degrees around y-axis
        rotate_with_delay([0, 0, 5])  # Rotate 5 degrees around z-axis
        rotate_with_delay([-5, 0, 0])  # Rotate back -5 degrees around x-axis
        rotate_with_delay([0, -5, 0])  # Rotate back -5 degrees around y-axis
        rotate_with_delay([0, 0, -5])  # Rotate back -5 degrees around z-axis

        loop_num += 1
        if loop_num % 10 == 0:
            print(f"Loop {loop_num} complete")
            current_pose = arm._state_client.get_ee_pose()
            current_rotation = Rotation.from_quat(current_pose[3:])
            rotation_diff = initial_rotation.inv() * current_rotation
            euler_diff = rotation_diff.as_euler('xyz', degrees=True)
            total_rotation_error = np.linalg.norm(euler_diff)
            
            print(f"Orientation error after {loop_num} loops:")
            print(f"  Roll (X): {euler_diff[0]:.6f} degrees")
            print(f"  Pitch (Y): {euler_diff[1]:.6f} degrees")
            print(f"  Yaw (Z): {euler_diff[2]:.6f} degrees")
            print(f"  Total rotation error: {total_rotation_error:.6f} degrees")
        
        if loop_num % 10 == 0:
            if input("Press 'q' to stop, or any other key to continue: ").lower() == 'q':
                break
    
    print("Cartesian Orientation Error Test Done")
    rclpy.shutdown()


if __name__ == "__main__":
    cartesian_position_error_test()
    cartesian_orientation_error_test()
