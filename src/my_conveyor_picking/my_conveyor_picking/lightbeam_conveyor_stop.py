"""Lightbeam sensor conveyor stop node.

Stops the conveyor belt when the lightbeam sensor detects an object
(depth value less than configured threshold) and sends box pose to
palletizing action server.
"""

import json

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from isaac_ros2_messages.srv import GetPrimAttribute, SetPrimAttribute
from palletizing_interfaces.action import Palletizing

from my_conveyor_picking.helper_functions.transformations import compute_relative_pose
from my_conveyor_picking.world_transform_resolver import WorldTransformResolver


class LightbeamConveyorStop(Node):
    """ROS2 node that stops conveyor when lightbeam detects an object."""

    DEFAULT_PRIM_PATHS = [
        "/World/ConveyorBelts/ConveyorTrack/ConveyorBeltGraph",
        "/World/ConveyorBelts/ConveyorTrack_01/ConveyorBeltGraph",
    ]
    DEFAULT_DEPTH_THRESHOLD = 1.0
    DEFAULT_TOPIC = "/lightbeam_hit"
    DEFAULT_INITIAL_VELOCITY = 0.4
    CONVEYOR_VELOCITY_OFF = 0.0
    DEFAULT_BOX_PRIM_PATH = "/World/Cube"
    DEFAULT_IK_BASE_LINK = "/World/Robot/ur5/ur5_robot/base_link_inertia"

    # Velocity check constants
    VELOCITY_THRESHOLD = 0.01  # m/s - box considered stopped below this
    VELOCITY_CHECK_INTERVAL = 0.1  # seconds between velocity checks
    VELOCITY_CHECK_TIMEOUT = 3.0  # seconds before giving up

    def __init__(self):
        """Initialize the lightbeam conveyor stop node."""
        super().__init__("lightbeam_conveyor_stop")

        # Conveyor parameters
        self.declare_parameter("conveyor.prim_paths", self.DEFAULT_PRIM_PATHS)
        self.declare_parameter(
            "conveyor.initial_velocity", self.DEFAULT_INITIAL_VELOCITY
        )

        # Lightbeam parameters
        self.declare_parameter(
            "lightbeam.depth_threshold", self.DEFAULT_DEPTH_THRESHOLD
        )
        self.declare_parameter("lightbeam.topic", self.DEFAULT_TOPIC)

        # Box and robot parameters
        self.declare_parameter("box.prim_paths", [self.DEFAULT_BOX_PRIM_PATH])
        self.declare_parameter("ik_base_link", self.DEFAULT_IK_BASE_LINK)

        # Get parameter values
        self._prim_paths = self.get_parameter("conveyor.prim_paths").value
        self._initial_velocity = self.get_parameter("conveyor.initial_velocity").value
        self._depth_threshold = self.get_parameter("lightbeam.depth_threshold").value
        self._topic = self.get_parameter("lightbeam.topic").value
        self._box_prim_paths = self.get_parameter("box.prim_paths").value
        self._current_box_index = 0
        self._ik_base_link = self.get_parameter("ik_base_link").value

        # Service clients
        self._set_prim_client = self.create_client(
            SetPrimAttribute, "/set_prim_attribute"
        )
        self._get_prim_client = self.create_client(
            GetPrimAttribute, "/get_prim_attribute"
        )

        # World transform resolver
        self._transform_resolver = WorldTransformResolver(self, self._get_prim_client)

        # Action client for palletizing
        self._palletizing_client = ActionClient(
            self, Palletizing, "palletizing"
        )

        # Subscription for lightbeam data
        self._subscription = self.create_subscription(
            Float32MultiArray,
            self._topic,
            self._lightbeam_callback,
            10,
        )

        # State tracking
        self._object_detected = False
        self._service_available = False
        self._initial_velocity_set = False
        self._palletizing_in_progress = False

        # World transforms (populated by resolver)
        self._base_link_world_pos = None
        self._base_link_world_orient = None
        self._box_world_pos = None
        self._box_world_orient = None

        # Velocity check state
        self._velocity_check_timer = None
        self._velocity_check_start_time = None

        # Timer to set initial velocity when service becomes available
        self._startup_timer = self.create_timer(0.5, self._check_service_and_start)

        self.get_logger().info("Lightbeam Conveyor Stop initialized")
        self.get_logger().info(f"Initial velocity: {self._initial_velocity}m/s")
        self.get_logger().info(f"Depth threshold: {self._depth_threshold}m")
        self.get_logger().info(f"Listening on: {self._topic}")
        self.get_logger().info(f"Controlling {len(self._prim_paths)} conveyors")
        self.get_logger().info(f"Box prim paths: {len(self._box_prim_paths)} boxes configured")
        self.get_logger().info(f"IK base link: {self._ik_base_link}")

    @property
    def _box_prim_path(self) -> str:
        """Get current box prim path based on index.

        Returns:
            Current box prim path or None if all boxes processed.
        """
        if self._current_box_index < len(self._box_prim_paths):
            return self._box_prim_paths[self._current_box_index]
        return None

    def _check_service_and_start(self) -> None:
        """Check if service is available and set initial velocity."""
        if self._initial_velocity_set:
            self._startup_timer.cancel()
            return

        if self._set_prim_client.service_is_ready():
            self._service_available = True
            self.get_logger().info("Service /set_prim_attribute available")
            self._set_initial_velocity()
            self._startup_timer.cancel()

    def _lightbeam_callback(self, msg: Float32MultiArray) -> None:
        """Handle incoming lightbeam depth data.

        Args:
            msg: Float32MultiArray with depth values for each ray.
        """
        if not self._service_available:
            if self._set_prim_client.service_is_ready():
                self._service_available = True
                self.get_logger().info("Service /set_prim_attribute available")
                self._set_initial_velocity()
            else:
                return

        object_detected = any(
            depth < self._depth_threshold for depth in msg.data
        )

        if object_detected and not self._object_detected:
            self._object_detected = True
            self._stop_conveyor()
            self.get_logger().info("Object detected! Stopping conveyor.")
            self._start_velocity_check()

        elif not object_detected and self._object_detected:
            self._object_detected = False
            self.get_logger().info("Object cleared.")

    def _set_initial_velocity(self) -> None:
        """Set initial velocity for all conveyor belts when simulation starts."""
        if self._initial_velocity_set:
            return

        for prim_path in self._prim_paths:
            request = SetPrimAttribute.Request()
            request.path = prim_path
            request.attribute = "graph:variable:Velocity"
            request.value = str(self._initial_velocity)
            self._set_prim_client.call_async(request)

        self._initial_velocity_set = True
        self.get_logger().info(
            f"Conveyor started with velocity: {self._initial_velocity}m/s"
        )

    def _stop_conveyor(self) -> None:
        """Stop all conveyor belts."""
        for prim_path in self._prim_paths:
            request = SetPrimAttribute.Request()
            request.path = prim_path
            request.attribute = "graph:variable:Velocity"
            request.value = str(self.CONVEYOR_VELOCITY_OFF)
            self._set_prim_client.call_async(request)

    def _restart_conveyor(self) -> None:
        """Restart all conveyor belts after palletizing."""
        for prim_path in self._prim_paths:
            request = SetPrimAttribute.Request()
            request.path = prim_path
            request.attribute = "graph:variable:Velocity"
            request.value = str(self._initial_velocity)
            self._set_prim_client.call_async(request)
        self.get_logger().info(
            f"Conveyor restarted: {self._initial_velocity}m/s"
        )

    def _start_velocity_check(self) -> None:
        """Start timer to check box velocity before getting pose."""
        if self._palletizing_in_progress:
            self.get_logger().warn("Palletizing already in progress, skipping.")
            return

        if not self._get_prim_client.service_is_ready():
            self.get_logger().warn("GetPrimAttribute service not available")
            return

        self._palletizing_in_progress = True
        self._velocity_check_start_time = self.get_clock().now()

        self.get_logger().info("Waiting for box to stop...")

        # Start timer to check velocity periodically
        self._velocity_check_timer = self.create_timer(
            self.VELOCITY_CHECK_INTERVAL,
            self._check_box_velocity
        )

    def _check_box_velocity(self) -> None:
        """Check if box has stopped moving."""
        elapsed = (self.get_clock().now() - self._velocity_check_start_time).nanoseconds / 1e9
        if elapsed > self.VELOCITY_CHECK_TIMEOUT:
            self.get_logger().error(
                f"Timeout waiting for box to stop after {self.VELOCITY_CHECK_TIMEOUT}s"
            )
            self._cancel_velocity_check()
            self._palletizing_in_progress = False
            return

        # Request velocity from prim service
        velocity_request = GetPrimAttribute.Request()
        velocity_request.path = self._box_prim_path
        velocity_request.attribute = "physics:velocity"

        velocity_future = self._get_prim_client.call_async(velocity_request)
        velocity_future.add_done_callback(self._on_velocity_response)

    def _on_velocity_response(self, future) -> None:
        """Handle velocity response."""
        try:
            result = future.result()
            if not result.success:
                self.get_logger().warn(f"Failed to get velocity: {result.message}")
                return

            velocity = json.loads(result.value)
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)

            if speed < self.VELOCITY_THRESHOLD:
                self.get_logger().info(f"Box stopped (velocity: {speed:.4f} m/s)")
                self._cancel_velocity_check()
                self._resolve_transforms()
            else:
                self.get_logger().debug(f"Box still moving: {speed:.4f} m/s")

        except Exception as e:
            self.get_logger().error(f"Error checking velocity: {e}")

    def _cancel_velocity_check(self) -> None:
        """Cancel the velocity check timer."""
        if self._velocity_check_timer is not None:
            self._velocity_check_timer.cancel()
            self._velocity_check_timer = None

    def _resolve_transforms(self) -> None:
        """Resolve world transforms for base_link and box using hierarchy traversal."""
        self.get_logger().info("Resolving base_link world transform...")
        self._transform_resolver.resolve(
            self._ik_base_link,
            self._on_base_link_resolved
        )

    def _on_base_link_resolved(self, position: list, orientation: list) -> None:
        """Handle base_link world transform resolution.

        Args:
            position: World position [x, y, z] in meters.
            orientation: World orientation [x, y, z, w] quaternion.
        """
        self._base_link_world_pos = position
        self._base_link_world_orient = orientation

        self.get_logger().info("Resolving box world transform...")
        self._transform_resolver.resolve(
            self._box_prim_path,
            self._on_box_resolved
        )

    def _on_box_resolved(self, position: list, orientation: list) -> None:
        """Handle box world transform resolution.

        Args:
            position: World position [x, y, z] in meters.
            orientation: World orientation [x, y, z, w] quaternion.
        """
        self._box_world_pos = position
        self._box_world_orient = orientation

        # Compute relative pose and send goal
        base_pos = np.array(self._base_link_world_pos)
        base_orient = np.array(self._base_link_world_orient)
        box_pos = np.array(self._box_world_pos)
        box_orient = np.array(self._box_world_orient)

        rel_pos, rel_orient = compute_relative_pose(
            base_pos, base_orient, box_pos, box_orient
        )

        self.get_logger().info(f"Relative position: {rel_pos.tolist()}")
        self.get_logger().info(f"Relative orientation: {rel_orient.tolist()}")

        self._send_palletizing_goal(rel_pos.tolist(), rel_orient.tolist())

    def _send_palletizing_goal(self, position: list, orientation: list) -> None:
        """Send palletizing goal with computed relative pose.

        Args:
            position: Box position [x, y, z] relative to base_link in meters.
            orientation: Box orientation [x, y, z, w] relative to base_link.
        """
        if not self._palletizing_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warn("Palletizing action server not available.")
            self._palletizing_in_progress = False
            return

        goal = Palletizing.Goal()

        # Box start pose (relative to robot base_link, in meters)
        goal.box_start_pose.position.x = position[0]
        goal.box_start_pose.position.y = position[1]
        goal.box_start_pose.position.z = position[2]
        goal.box_start_pose.orientation.x = orientation[0]
        goal.box_start_pose.orientation.y = orientation[1]
        goal.box_start_pose.orientation.z = orientation[2]
        goal.box_start_pose.orientation.w = orientation[3]

        # Box target pose (placeholder)
        goal.box_target_pose.position.x = 0.0
        goal.box_target_pose.position.y = 0.0
        goal.box_target_pose.position.z = 0.0
        goal.box_target_pose.orientation.x = 0.0
        goal.box_target_pose.orientation.y = 0.0
        goal.box_target_pose.orientation.z = 0.0
        goal.box_target_pose.orientation.w = 1.0

        self.get_logger().info("Sending palletizing goal...")
        send_goal_future = self._palletizing_client.send_goal_async(
            goal, feedback_callback=self._palletizing_feedback_callback
        )
        send_goal_future.add_done_callback(self._palletizing_goal_response_callback)

    def _palletizing_goal_response_callback(self, future) -> None:
        """Handle palletizing goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Palletizing goal rejected")
            self._palletizing_in_progress = False
            return

        self.get_logger().info("Palletizing goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._palletizing_result_callback)

    def _palletizing_result_callback(self, future) -> None:
        """Handle palletizing result."""
        print("========== RESULT CALLBACK REACHED ==========")
        result = future.result().result
        self.get_logger().info(f"Palletizing result: {result.error_string}")

        # Advance to next box
        self._current_box_index += 1

        # Check if more boxes to pick
        if self._current_box_index < len(self._box_prim_paths):
            self.get_logger().info(
                f"Next box: {self._current_box_index + 1}/{len(self._box_prim_paths)}"
            )
            # Reset detection state and restart conveyor
            self._object_detected = False
            print(">>>>>> CALLING RESTART CONVEYOR <<<<<<")
            self._restart_conveyor()
        else:
            self.get_logger().info("All boxes picked! Pile complete.")

        self._palletizing_in_progress = False

    def _palletizing_feedback_callback(self, feedback_msg) -> None:
        """Handle palletizing feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().debug(
            f"Palletizing feedback - EE position: "
            f"({feedback.end_effector_pose.position.x:.3f}, "
            f"{feedback.end_effector_pose.position.y:.3f}, "
            f"{feedback.end_effector_pose.position.z:.3f})"
        )


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = LightbeamConveyorStop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
