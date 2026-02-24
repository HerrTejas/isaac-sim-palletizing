#!/usr/bin/python3
"""Inverse kinematics of the UR5 robot."""
import cmath
from math import acos, atan2, cos, pi, sin, sqrt

import numpy as np
import numpy.linalg as linalg

from .transformations import (htm_rotation_around_x, htm_rotation_around_y,
                              htm_rotation_around_z, htm_translation)

MAT = np.matrix


class InverseKinematics:
    """UR5 Inverse Kinematics solver with wrist angle optimization.

    Tracks previous joint angles to minimize wrist rotation between
    consecutive movements.

    Class Attributes
    ----------------
    TRANSL_PARAM_Z : np.array
        DH parameter d (translation along Z) for each joint.
    TRANSL_PARAM_X : np.array
        DH parameter a (translation along X) for each joint.
    ROT_PARAM_X : np.array
        DH parameter alpha (rotation around X) for each joint.

    Example
    -------
    >>> ik = InverseKinematics(solution_index=5)
    >>> angles = ik.solve(desired_pose)
    >>> # For next movement in sequence:
    >>> angles = ik.solve(next_pose)  # Wrist automatically normalized
    >>> # Start new sequence:
    >>> ik.reset()
    """

    TRANSL_PARAM_Z = np.array([0.089159, 0, 0, 0, 0.10915, 0.09465, 0.0823])
    TRANSL_PARAM_X = np.array([0, 0, -0.425, -0.39225, 0, 0, 0])
    ROT_PARAM_X = np.array([0, np.pi/2, 0, 0, 0, 0, -np.pi/2])

    def __init__(
        self,
        solution_index: int = 5,
        ee_offset_position: list = None,
        ee_offset_orientation: list = None
    ):
        """Initialize IK solver.

        Parameters
        ----------
        solution_index : int
            IK solution configuration index (0-7).
            Default 5 represents elbow up, wrist up configuration.
        ee_offset_position : list, optional
            [x, y, z] position offset from wrist_3_link to gripper tip in meters.
        ee_offset_orientation : list, optional
            [roll, pitch, yaw] orientation offset in degrees.
        """
        self.solution_index = solution_index
        self.previous_angles = None
        self.ee_offset_htm = None

        if ee_offset_position is not None and ee_offset_orientation is not None:
            self.ee_offset_htm = self._build_ee_offset_htm(
                ee_offset_position, ee_offset_orientation
            )

    def solve(
        self,
        desired_pose: np.array,
        print_debug: bool = False
    ) -> np.ndarray:
        """Compute IK solution for desired pose.

        Parameters
        ----------
        desired_pose : np.array
            4x4 HTM of desired pose in base_link_inertia frame.
            If ee_offset was set, this should be the gripper/tool pose.
            Otherwise, this should be the wrist_3_link pose.
        print_debug : bool
            Print all 8 solutions if True.

        Returns
        -------
        np.ndarray
            6 joint angles in radians.
        """
        # Transform gripper pose to wrist_3_link pose if offset is set
        if self.ee_offset_htm is not None:
            wrist_pose = desired_pose @ linalg.inv(self.ee_offset_htm)
        else:
            wrist_pose = desired_pose

        solutions = self._compute_all_solutions(wrist_pose)

        if print_debug:
            solution_deg = np.degrees(solutions).astype(int)
            print(f"IK Solutions [degrees]:\n{solution_deg}")

        result = np.array(solutions.real[:, self.solution_index]).reshape(6)

        if self.previous_angles is not None:
            result = self._normalize_wrist(result)

        self.previous_angles = result.copy()
        return result

    def reset(self):
        """Reset state for new movement sequence."""
        self.previous_angles = None

    def set_ee_offset(
        self,
        ee_offset_position: list = None,
        ee_offset_orientation: list = None
    ):
        """Set or clear the end-effector offset.

        Parameters
        ----------
        ee_offset_position : list, optional
            [x, y, z] position offset from wrist_3_link to gripper tip in meters.
            Pass None to clear the offset.
        ee_offset_orientation : list, optional
            [roll, pitch, yaw] orientation offset in degrees.
            Pass None to clear the offset.
        """
        if ee_offset_position is not None and ee_offset_orientation is not None:
            self.ee_offset_htm = self._build_ee_offset_htm(
                ee_offset_position, ee_offset_orientation
            )
        else:
            self.ee_offset_htm = None

    def _build_ee_offset_htm(
        self,
        position: list,
        orientation: list
    ) -> np.matrix:
        """Build HTM for end-effector offset from wrist_3_link.

        Parameters
        ----------
        position : list
            [x, y, z] position in meters.
        orientation : list
            [roll, pitch, yaw] in degrees.

        Returns
        -------
        np.matrix
            4x4 homogeneous transformation matrix.
        """
        from .transformations import get_desired_pose_htm
        return get_desired_pose_htm(
            position=np.array(position),
            roll=orientation[0],
            pitch=orientation[1],
            yaw=orientation[2]
        )

    # ==================== HTM Methods ====================

    def _htm_01(self, theta1: float) -> np.matrix:
        """HTM from base_link_inertia to shoulder_link."""
        translation_z = htm_translation([0, 0, self.TRANSL_PARAM_Z[0]])
        rotation_z = htm_rotation_around_z(theta1)
        # We post multiply rotation_z because
        # shoulder_link is rotating around shoulder_link z axis
        # and not around base_link_inertia z axis
        return translation_z * rotation_z

    def _htm_12(self, theta2: float) -> np.matrix:
        """HTM from shoulder_link to upper_arm_link."""
        rotation_x = htm_rotation_around_x(self.ROT_PARAM_X[1])
        rotation_z = htm_rotation_around_z(theta2)
        # We post multiply rotation_z because
        # upper_arm_link is rotating around upper_arm_link z axis
        # and not around shoulder_link z axis
        return rotation_x * rotation_z

    def _htm_23(self, theta3: float) -> np.matrix:
        """HTM from upper_arm_link to forearm_link."""
        translation_x = htm_translation([self.TRANSL_PARAM_X[2], 0, 0])
        rotation_z = htm_rotation_around_z(theta3)
        return translation_x * rotation_z

    def _htm_34(self) -> np.matrix:
        """HTM from forearm_link to forearm_link_x."""
        return htm_translation([self.TRANSL_PARAM_X[3], 0, 0])

    def _htm_45(self, theta4: float) -> np.matrix:
        """HTM from forearm_link_x to wrist_1_link."""
        translation_z = htm_translation([0, 0, self.TRANSL_PARAM_Z[4]])
        rotation_z = htm_rotation_around_z(theta4)
        return translation_z * rotation_z

    def _htm_56(self, theta5: float) -> np.matrix:
        """HTM from wrist_1_link to wrist_2_link."""
        rotation_x = htm_rotation_around_x(self.ROT_PARAM_X[5])
        translation_z = htm_translation([0, 0, self.TRANSL_PARAM_X[5]])
        rotation_z = htm_rotation_around_z(theta5)
        return rotation_x * translation_z * rotation_z

    def _htm_67(self, theta6: float) -> np.matrix:
        """HTM from wrist_2_link to wrist_3_link."""
        rotation_x = htm_rotation_around_x(self.ROT_PARAM_X[-1])
        translation_z = htm_translation([0, 0, self.TRANSL_PARAM_Z[-1]])
        rotation_z = htm_rotation_around_z(theta6)
        return rotation_x * translation_z * rotation_z

    # ==================== Theta Calculation Methods ====================

    def _compute_all_solutions(self, desired_pose: np.array) -> np.matrix:
        """Compute all 8 IK solutions."""
        solutions = MAT(np.zeros((6, 8)))
        solutions = self._get_theta1(solutions, desired_pose)
        solutions = self._get_theta5(solutions, desired_pose)
        solutions = self._get_theta6(solutions, desired_pose)
        solutions = self._get_theta3(solutions, desired_pose)
        solutions = self._get_theta2(solutions, desired_pose)
        solutions = self._get_theta4(solutions, desired_pose)
        return solutions

    def _get_theta1(
        self,
        solutions: np.array,
        desired_pose_07: np.array
    ) -> np.matrix:
        """Calculate theta1 angles for all configurations."""
        # position_link_89 is a column vector with the position of
        # the wrist_2_link in respect to the base_link_inertia
        position_link_89 = \
            desired_pose_07 * MAT([0, 0, -self.TRANSL_PARAM_Z[6], 1]).T \
            - MAT([0, 0, 0, 1]).T

        # beta is the angle between the projection in the xy plane of the
        # wrist_2_link and the x axis of the base_link_inertia
        beta = atan2(position_link_89[1, 0], position_link_89[0, 0])

        # Magnitude of the projection of the wrist_2_link in the xy plane
        position_wrist_2_xy = sqrt(
            position_link_89[1, 0]**2 + position_link_89[0, 0]**2)

        if self.TRANSL_PARAM_Z[4] > position_wrist_2_xy:
            raise ValueError(
                "d4 cannot be higher than position_wrist_2_xy. "
                "No solution for theta1")

        gamma = acos(self.TRANSL_PARAM_Z[4] / position_wrist_2_xy)

        # Shoulder left or right (depends on shoulder_lift_joint and elbow_joint)
        # Be careful, shoulder right or left does not mean the direction of the
        # shoulder joint. It depends on shoulder_lift_joint angle and elbow_joint
        # angle. In other words, there is no rule to know if the shoulder is right
        # or left. It depends on the other joints.
        solutions[0, 0:4] = pi/2 + beta + gamma
        # For the Laboratory of Robotics at UFBA, keep the z axis of upper_arm_link
        # pointing to the direction opposite to the laboratory entrance or
        # (pi/2 + beta - gamma)
        solutions[0, 4:8] = pi/2 + beta - gamma

        return solutions

    def _get_theta5(
        self,
        solutions: np.array,
        desired_pose_07: np.array
    ) -> np.matrix:
        """Calculate theta5 angles for all configurations.

        Recommended interval for theta5 at the Laboratory of Robotics at UFBA
        if the task is to pick an object from the table or inside the printer:
        [0, pi]
        """
        wrist_up_or_down_configs = [0, 4]
        for config in wrist_up_or_down_configs:
            # Theta1 can be used since it is already calculated
            htm_link_01 = self._htm_01(solutions[0, config])
            # We assume that theta2 is still zero since it is not calculated yet
            htm_link_12 = self._htm_12(0)
            htm_link_02 = htm_link_01 * htm_link_12
            htm_link_20 = linalg.inv(htm_link_02)
            htm_link_27 = htm_link_20 * desired_pose_07

            acos_num = htm_link_27[2, 3] - self.TRANSL_PARAM_Z[4]
            acos_den = self.TRANSL_PARAM_Z[6]

            if acos_num > acos_den:
                raise ValueError(
                    "P16z - d4 cannot be higher than d6. "
                    "The z-axis of wrist_3_link cannot be parallel to the "
                    "wrist_2_link, wrist_1_link, forearm_link and "
                    "upper_arm_link z axis.")

            theta_5 = acos(acos_num / acos_den)

            # NOTE: For UR5 at the Laboratory of Robotics at UFBA
            # If you are picking an object from the table or inside the printer
            # use +theta_5
            solutions[4, config:config+2] = theta_5
            solutions[4, config+2:config+4] = -theta_5

        return solutions

    def _get_theta6(
        self,
        solutions: np.array,
        desired_pose_07: np.array
    ) -> np.matrix:
        """Calculate theta6 angles for all configurations.

        Range of theta6: [-pi, pi]
        Note: theta6 is not well-defined when sin(theta5) = 0
        or when T16(1,3), T16(2,3) = 0.
        """
        configs = [0, 2, 4, 6]
        for config in configs:
            htm_link_01 = self._htm_01(solutions[0, config])
            htm_link_12 = self._htm_12(0)
            htm_link_02 = htm_link_01 * htm_link_12
            htm_link_20 = linalg.inv(htm_link_02)
            htm_link_27 = htm_link_20 * desired_pose_07
            htm_link_72 = linalg.inv(htm_link_27)

            theta5 = solutions[4, config]
            sin_theta5 = sin(theta5)

            theta6 = atan2(
                -htm_link_72[1, 2] / sin_theta5,
                htm_link_72[0, 2] / sin_theta5)
            solutions[5, config:config+2] = theta6

        return solutions

    def _get_theta3(
        self,
        solutions: np.array,
        desired_pose_07: np.array
    ) -> np.matrix:
        """Calculate theta3 angles for all configurations."""
        configs = [0, 2, 4, 6]
        for config in configs:
            theta1 = solutions[0, config]
            htm_link_01 = self._htm_01(theta1)
            htm_link_12 = self._htm_12(0)
            htm_link_02 = htm_link_01 * htm_link_12
            htm_link_20 = linalg.inv(htm_link_02)
            htm_link_27 = htm_link_20 * desired_pose_07

            theta6 = solutions[5, config]
            htm_link_67 = self._htm_67(theta6)
            theta5 = solutions[4, config]
            htm_link_56 = self._htm_56(theta5)

            htm_link_25 = htm_link_27 * linalg.inv(htm_link_56 * htm_link_67)

            position_24 = htm_link_25 * \
                MAT([0, 0, -self.TRANSL_PARAM_Z[4], 1]).T - MAT([0, 0, 0, 1]).T

            theta3 = cmath.acos((
                linalg.norm(position_24)**2
                - self.TRANSL_PARAM_X[2]**2 - self.TRANSL_PARAM_X[3]**2) /
                (2 * self.TRANSL_PARAM_X[2] * self.TRANSL_PARAM_X[3]))
            solutions[2, config] = theta3.real
            solutions[2, config+1] = -theta3.real

        return solutions

    def _get_theta2(
        self,
        solutions: np.array,
        desired_pose_07: np.array
    ) -> np.matrix:
        """Calculate theta2 angles for all configurations."""
        configs = [0, 1, 2, 3, 4, 5, 6, 7]
        for config in configs:
            theta1 = solutions[0, config]
            htm_link_01 = self._htm_01(theta1)
            htm_link_12 = self._htm_12(0)
            htm_link_02 = htm_link_01 * htm_link_12
            htm_link_20 = linalg.inv(htm_link_02)
            htm_link_27 = htm_link_20 * desired_pose_07

            theta6 = solutions[5, config]
            htm_link_67 = self._htm_67(theta6)
            theta5 = solutions[4, config]
            htm_link_56 = self._htm_56(theta5)

            htm_link_25 = htm_link_27 * linalg.inv(htm_link_56 * htm_link_67)

            position_24 = htm_link_25 * \
                MAT([0, 0, -self.TRANSL_PARAM_Z[4], 1]).T - MAT([0, 0, 0, 1]).T

            gamma = atan2(position_24[1, 0], position_24[0, 0])
            theta3 = solutions[2, config]
            beta = atan2(
                self.TRANSL_PARAM_X[3] * sin(theta3),
                self.TRANSL_PARAM_X[2] + self.TRANSL_PARAM_X[3] * cos(theta3))
            theta2 = gamma - beta
            solutions[1, config] = theta2

        return solutions

    def _get_theta4(
        self,
        solutions: np.array,
        desired_pose_07: np.array
    ) -> np.matrix:
        """Calculate theta4 angles for all configurations."""
        configs = [0, 1, 2, 3, 4, 5, 6, 7]
        for config in configs:
            theta1 = solutions[0, config]
            htm_link_01 = self._htm_01(theta1)
            theta2 = solutions[1, config]
            htm_link_12 = self._htm_12(theta2)
            htm_link_02 = htm_link_01 * htm_link_12
            htm_link_20 = linalg.inv(htm_link_02)
            htm_link_27 = htm_link_20 * desired_pose_07

            theta6 = solutions[5, config]
            htm_link_67 = self._htm_67(theta6)
            theta5 = solutions[4, config]
            htm_link_56 = self._htm_56(theta5)

            htm_link_25 = htm_link_27 * linalg.inv(htm_link_56 * htm_link_67)

            htm_link_34 = self._htm_34()
            theta3 = solutions[2, config]
            htm_link_23 = self._htm_23(theta3)
            htm_link_45 = linalg.inv(htm_link_34) * \
                linalg.inv(htm_link_23) * htm_link_25
            theta4 = atan2(htm_link_45[1, 0], htm_link_45[0, 0])

            solutions[3, config] = theta4

        return solutions

    def _normalize_wrist(self, angles: np.ndarray) -> np.ndarray:
        """Normalize wrist_3 angle to minimize rotation."""
        diff = angles[5] - self.previous_angles[5]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        angles[5] = self.previous_angles[5] + diff
        return angles

    def debug_htm_matrices(self, angles: np.array):
        """Print the homogeneous transformation matrices for debugging."""
        print("---------------")
        htm_01 = self._htm_01(angles[0])
        print(f"htm_01: \n{np.round(htm_01, 3)}")

        htm_12 = self._htm_12(angles[1])
        print(f"htm_12: \n{np.round(htm_12, 3)}")

        htm_02 = htm_01 * htm_12
        print(f"htm_02: \n{np.round(htm_02, 3)}")

        htm_20 = linalg.inv(htm_02)
        print(f"htm_20: \n{np.round(htm_20, 3)}")

        htm_23 = self._htm_23(angles[2])
        print(f"htm_23: \n{np.round(htm_23, 3)}")

        htm_34 = self._htm_34()
        print(f"htm_34: \n{np.round(htm_34, 3)}")

        htm_45 = self._htm_45(angles[3])
        print(f"htm_45: \n{np.round(htm_45, 3)}")

        htm_35 = htm_34 * htm_45
        print(f"htm_35: \n{np.round(htm_35, 3)}")

        htm_56 = self._htm_56(angles[4])
        print(f"htm_56: \n{np.round(htm_56, 3)}")

        htm_67 = self._htm_67(angles[5])
        print(f"htm_67: \n{np.round(htm_67, 3)}")

        htm_07 = htm_02 * htm_23 * htm_35 * htm_56 * htm_67
        print(f"htm_07: \n{np.round(htm_07, 3)}")


def main():
    """Main function for testing."""
    roll_angle_deg = 0
    pitch_angle_deg = 90
    yaw_angle_deg = 0

    rotation_around_x = htm_rotation_around_x(np.radians(roll_angle_deg))
    rotation_around_y = htm_rotation_around_y(np.radians(pitch_angle_deg))
    rotation_around_z = htm_rotation_around_z(np.radians(yaw_angle_deg))
    desired_pose = rotation_around_x * rotation_around_y * rotation_around_z

    position_vector = np.array([[0.5, 0.0, 0.5]]).T
    desired_pose[:3, -1] = position_vector
    print(f"desired_pose: \n{desired_pose}")

    ik = InverseKinematics(solution_index=5)
    joint_angles = ik.solve(desired_pose)
    print(f"Joint angles: \n{joint_angles}")


if __name__ == "__main__":
    main()
