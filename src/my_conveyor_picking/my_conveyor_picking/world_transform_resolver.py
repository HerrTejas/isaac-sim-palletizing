"""World transform resolver for USD prim paths.

Traverses the prim hierarchy and chains transforms to compute
world position and orientation for any prim path.
"""

import json
from typing import Callable, Optional

import numpy as np
from rclpy.node import Node
from isaac_ros2_messages.srv import GetPrimAttribute

from my_conveyor_picking.helper_functions.transformations import (
    quaternion_multiply,
    quaternion_rotate,
    quaternion_to_euler_deg,
)


class WorldTransformResolver:
    """Resolves world transforms by traversing USD prim hierarchy."""

    def __init__(self, node: Node, service_client):
        """Initialize the resolver.

        Args:
            node: ROS2 node for logging.
            service_client: GetPrimAttribute service client.
        """
        self._node = node
        self._client = service_client

        # State for current resolution
        self._prim_levels: list[str] = []
        self._current_level: int = 0
        self._accumulated_position: np.ndarray = np.zeros(3)
        self._accumulated_orientation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self._accumulated_scale: np.ndarray = np.ones(3)
        self._callback: Optional[Callable] = None

        # Current level data
        self._current_translate: Optional[np.ndarray] = None
        self._current_orient: Optional[np.ndarray] = None
        self._current_scale: Optional[np.ndarray] = None

    def resolve(self, prim_path: str, callback: Callable) -> None:
        """Resolve world transform for a prim path.

        Args:
            prim_path: Full USD prim path (e.g., /World/Robot/ur5/base_link).
            callback: Function to call with (position, orientation) when done.
                      Position is [x, y, z] in meters.
                      Orientation is [x, y, z, w] quaternion.
        """
        # Parse prim path into levels
        # /World/Robot/ur5 -> ['World', 'Robot', 'ur5']
        parts = [p for p in prim_path.split('/') if p]

        # Build cumulative paths: /World, /World/Robot, /World/Robot/ur5, etc.
        self._prim_levels = []
        for i in range(len(parts)):
            self._prim_levels.append('/' + '/'.join(parts[:i+1]))

        self._node.get_logger().info(
            f"Resolving world transform for: {prim_path}"
        )
        self._node.get_logger().info(f"  Levels: {self._prim_levels}")

        # Reset state
        self._current_level = 0
        self._accumulated_position = np.zeros(3)
        self._accumulated_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self._accumulated_scale = np.ones(3)
        self._callback = callback

        # Start fetching transforms
        self._fetch_level_translate()

    def _fetch_level_translate(self) -> None:
        """Fetch translate for current level."""
        if self._current_level >= len(self._prim_levels):
            # Done with all levels
            self._finish()
            return

        prim_path = self._prim_levels[self._current_level]
        request = GetPrimAttribute.Request()
        request.path = prim_path
        request.attribute = "xformOp:translate"

        future = self._client.call_async(request)
        future.add_done_callback(self._on_translate_response)

    def _on_translate_response(self, future) -> None:
        """Handle translate response."""
        try:
            result = future.result()
            if result.success:
                self._current_translate = np.array(json.loads(result.value))
            else:
                # No translate, use zero
                self._current_translate = np.zeros(3)

            # Fetch orientation
            self._fetch_level_orient()

        except Exception as e:
            self._node.get_logger().error(f"Error fetching translate: {e}")
            self._current_translate = np.zeros(3)
            self._fetch_level_orient()

    def _fetch_level_orient(self) -> None:
        """Fetch orientation for current level."""
        prim_path = self._prim_levels[self._current_level]
        request = GetPrimAttribute.Request()
        request.path = prim_path
        request.attribute = "xformOp:orient"

        future = self._client.call_async(request)
        future.add_done_callback(self._on_orient_response)

    def _on_orient_response(self, future) -> None:
        """Handle orientation response."""
        try:
            result = future.result()
            if result.success:
                # USD quaternion is [w, x, y, z], convert to [x, y, z, w]
                orient_wxyz = json.loads(result.value)
                self._current_orient = np.array([
                    orient_wxyz[1], orient_wxyz[2], orient_wxyz[3], orient_wxyz[0]
                ])
            else:
                # No orientation, use identity
                self._current_orient = np.array([0.0, 0.0, 0.0, 1.0])

            # Fetch scale
            self._fetch_level_scale()

        except Exception as e:
            self._node.get_logger().error(f"Error fetching orient: {e}")
            self._current_orient = np.array([0.0, 0.0, 0.0, 1.0])
            self._fetch_level_scale()

    def _fetch_level_scale(self) -> None:
        """Fetch scale for current level."""
        prim_path = self._prim_levels[self._current_level]
        request = GetPrimAttribute.Request()
        request.path = prim_path
        request.attribute = "xformOp:scale"

        future = self._client.call_async(request)
        future.add_done_callback(self._on_scale_response)

    def _on_scale_response(self, future) -> None:
        """Handle scale response."""
        try:
            result = future.result()
            if result.success:
                scale_val = json.loads(result.value)
                if isinstance(scale_val, list):
                    self._current_scale = np.array(scale_val)
                else:
                    # Uniform scale
                    self._current_scale = np.array([scale_val, scale_val, scale_val])
            else:
                # No scale, use 1
                self._current_scale = np.ones(3)

        except Exception as e:
            self._node.get_logger().error(f"Error fetching scale: {e}")
            self._current_scale = np.ones(3)

        # Apply this level's transform and move to next
        self._apply_level_transform()

    def _apply_level_transform(self) -> None:
        """Apply current level's transform to accumulated transform."""
        prim_path = self._prim_levels[self._current_level]

        # Scale the local translation by accumulated scale
        scaled_translate = self._current_translate * self._accumulated_scale

        # Rotate scaled translation by accumulated orientation
        rotated_translate = quaternion_rotate(
            self._accumulated_orientation, scaled_translate
        )

        # Add to accumulated position
        self._accumulated_position = self._accumulated_position + rotated_translate

        # Combine orientations: accumulated * current
        self._accumulated_orientation = quaternion_multiply(
            self._accumulated_orientation, self._current_orient
        )

        # Combine scales (element-wise)
        self._accumulated_scale = self._accumulated_scale * self._current_scale

        self._node.get_logger().debug(
            f"  Level {prim_path}: translate={self._current_translate.tolist()}, "
            f"orient={self._current_orient.tolist()}, scale={self._current_scale.tolist()}"
        )

        # Move to next level
        self._current_level += 1
        self._fetch_level_translate()

    def _finish(self) -> None:
        """Complete resolution and call callback."""
        self._node.get_logger().info(
            f"  World position: {self._accumulated_position.tolist()}"
        )

        euler = quaternion_to_euler_deg(self._accumulated_orientation)
        self._node.get_logger().info(
            f"  World orientation (euler): roll={euler[0]:.2f}, "
            f"pitch={euler[1]:.2f}, yaw={euler[2]:.2f}"
        )

        if self._callback:
            self._callback(
                self._accumulated_position.tolist(),
                self._accumulated_orientation.tolist()
            )
