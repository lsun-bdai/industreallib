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
    
    print("Done")
    rclpy.shutdown()

if __name__ == "__main__":
    cartesian_position_error_test()