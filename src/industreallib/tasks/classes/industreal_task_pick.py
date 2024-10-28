# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Class definition for IndustRealTaskPick.

This script defines the IndustRealTaskPick class. The class gets the
observations that are specific to the Pick task and can also contain
other task-specific methods.
"""

# Third Party
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# NVIDIA
from industreallib.tasks.classes.industreal_task_base import IndustRealTaskBase


class IndustRealTaskPick(IndustRealTaskBase):
    """Defines class for Pick task."""

    def __init__(self, args, task_instance_config, in_sequence):
        """Initializes the superclass."""
        super().__init__(
            args=args, task_instance_config=task_instance_config, in_sequence=in_sequence
        )

    def _get_observations(self, goal_pos, goal_ori_mat, franka_arm):
        """Gets the robot state from frankapy. Extracts the observations."""
        # NOTE: The position and orientation observations should be in the same
        # coordinate frame as during training, and the orientation observations
        # should have the same representations as during training. In Factory and
        # IndustReal, the position and orientation observations are typically in
        # the robot base frame, and the orientation observations are typically
        # represented as quaternions (x, y, z, w).

        curr_state = {}
        curr_joint_angles = franka_arm.get_joint_positions()
        curr_pos = franka_arm.get_ee_pose()[:3]
        curr_ori_quat = franka_arm.get_ee_pose()[3:7] # xyzw
        goal_ori_quat = Rotation.from_matrix(goal_ori_mat).as_quat()
        observations = (
            torch.from_numpy(
                np.hstack(
                    [
                        curr_joint_angles,
                        curr_pos,
                        curr_ori_quat,
                        goal_pos,
                        goal_ori_quat,
                    ]
                )
            )
            .to(torch.float32)
            .to(self._device)
        )
        curr_state["joint_angles"] = curr_joint_angles
        curr_state["ee_pose"] = np.concatenate([curr_pos, curr_ori_quat])
        curr_state["ee_ori_mat"] = Rotation.from_quat(curr_ori_quat).as_matrix()

        return observations, curr_state
