"""Transformation utilities for robotics applications.

Includes homogeneous transformation matrices (HTM) and quaternion operations.
"""

import math
from math import cos, sin

import numpy as np
import tf_transformations


def geometry_msg_pose_to_htm(geometry_msg):
    """Convert a geometry_msgs.transform message to a homogeneous transformation matrix.

    Parameters
    ----------
    geometry_msg: geometry_msgs.transform
        Pose message to convert

    Returns
    -------
    np.array:
        Homogeneous transformation matrix
    """
    position = geometry_msg.translation
    orientation = geometry_msg.rotation
    translation = np.array([position.x, position.y, position.z])
    rotation = np.array([orientation.x, orientation.y,
                         orientation.z, orientation.w])
    homogeneous_transformation = tf_transformations.quaternion_matrix(rotation)
    homogeneous_transformation[0:3, 3] = translation
    return homogeneous_transformation


def htm_rotation_around_x(angle: float) -> np.matrix:
    """Rotation matrix around x axis

    Parameters
    ----------
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.matrix
        Rotation

    """
    return np.matrix([[1, 0, 0, 0],
                     [0, cos(angle), -sin(angle), 0],
                     [0, sin(angle), cos(angle), 0],
                     [0, 0, 0, 1]])


def htm_rotation_around_y(angle: float) -> np.matrix:
    """Rotation matrix around y axis

    Parameters
    ----------
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.matrix
        Rotation

    """
    return np.matrix([[cos(angle), 0, sin(angle), 0],
                     [0, 1, 0, 0],
                     [-sin(angle), 0, cos(angle), 0],
                     [0, 0, 0, 1]])


def htm_rotation_around_z(angle: float) -> np.matrix:
    """Rotation matrix around z axis

    Parameters
    ----------
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.matrix
        Rotation

    """
    return np.matrix([[cos(angle), -sin(angle), 0, 0],
                     [sin(angle), cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def htm_translation(translation_vector: np.array) -> np.matrix:
    """Generate a homogeneous transformation matrix for translation

    Parameters
    ----------
    translation : np.array
        Translation vector [x, y, z]

    Returns
    -------
    np.matrix
        Translation

    """
    return np.matrix([[1, 0, 0, translation_vector[0]],
                     [0, 1, 0, translation_vector[1]],
                     [0, 0, 1, translation_vector[2]],
                     [0, 0, 0, 1]])


def get_desired_pose_htm(
    position: np.array,
    roll: float,
    pitch: float,
    yaw: float
):
    """Get the desired end effector pose in the base_link_inertia frame.

    Parameters
    ----------
    position : np.array. shape = (3,)
        Position of the end effector in the base_link_inertia frame
    roll : float
        Roll angle in degrees
    pitch : float
        Pitch angle in degrees
    yaw : float
        Yaw angle in degrees

    """
    rot_x = htm_rotation_around_x(np.radians(roll))
    rot_y = htm_rotation_around_y(np.radians(pitch))
    rot_z = htm_rotation_around_z(np.radians(yaw))
    desired_pose = rot_z * rot_y * rot_x

    # position vector is the desired position of the end effector
    # in the base_link_inertia frame
    position_vector = position.reshape(3, 1)

    # set the desired pose to have the same position as the position vector
    desired_pose[:3, -1] = position_vector

    return desired_pose


# =============================================================================
# Quaternion Operations
# =============================================================================


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        q1: First quaternion [x, y, z, w].
        q2: Second quaternion [x, y, z, w].

    Returns:
        Product quaternion [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternions).

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        Inverse quaternion [x, y, z, w].
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion.

    Args:
        q: Quaternion [x, y, z, w].
        v: Vector [x, y, z] to rotate.

    Returns:
        Rotated vector [x, y, z].
    """
    v_quat = np.array([v[0], v[1], v[2], 0.0])
    q_inv = quaternion_inverse(q)
    result = quaternion_multiply(quaternion_multiply(q, v_quat), q_inv)
    return result[:3]


def quaternion_to_euler_deg(q: np.ndarray) -> list:
    """Convert quaternion to Euler angles in degrees.

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        Euler angles [roll, pitch, yaw] in degrees.
    """
    x, y, z, w = q

    # Roll (rotation around X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation around Y)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (rotation around Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def compute_relative_pose(
    base_pos: np.ndarray,
    base_orient: np.ndarray,
    target_pos: np.ndarray,
    target_orient: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target pose relative to base frame.

    Transforms the target position and orientation from world frame
    to the base frame coordinate system.

    Args:
        base_pos: Base frame position [x, y, z] in world frame.
        base_orient: Base frame orientation [x, y, z, w] in world frame.
        target_pos: Target position [x, y, z] in world frame.
        target_orient: Target orientation [x, y, z, w] in world frame.

    Returns:
        Tuple of (relative_position, relative_orientation) where:
        - relative_position: Target position in base frame [x, y, z].
        - relative_orientation: Target orientation in base frame [x, y, z, w].
    """
    # Position difference in world frame
    pos_diff_world = target_pos - base_pos

    # Rotate position difference into base frame
    base_orient_inv = quaternion_inverse(base_orient)
    rel_position = quaternion_rotate(base_orient_inv, pos_diff_world)

    # Relative orientation: q_base_inv * q_target
    rel_orientation = quaternion_multiply(base_orient_inv, target_orient)

    return rel_position, rel_orientation
