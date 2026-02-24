#!/usr/bin/env python3
"""ROS2 node for UR5 box picking and palletizing coordination.

This node coordinates the palletizing process by:
1. Fetching box position/orientation from Isaac Sim via service calls
2. Sending pick-and-place goals to the UR5 palletizing action server
3. Notifying simulation when boxes are placed
"""

import json
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger

from isaac_ros2_messages.srv import GetPrimAttribute, SetPrimAttribute
from palletizing_interfaces.action import Palletizing
from my_conveyor_picking.helper_functions.load_ros_parameters import (
    get_ros_parameters
)


@dataclass
class PileConfig:
    """Configuration for pile generation.

    All positions are in the base_link_inertia frame.
    Spacing between box centers = (box_size + gap) * direction
    """

    starting_position: List[float]
    box_size: List[float]  # [x, y, z] dimensions
    x_gap: float
    y_gap: float
    z_gap: float
    x_direction: int  # -1 or 1
    y_direction: int  # -1 or 1
    z_direction: int  # -1 or 1
    x_count: int
    y_count: int
    z_count: int

    @property
    def max_boxes(self) -> int:
        """Total number of boxes in the pile."""
        return self.x_count * self.y_count * self.z_count

    def generate_positions(self) -> List[List[float]]:
        """Generate all target positions for boxes in the pile.

        Spacing formula: spacing = (box_size + gap) * direction
        - positive gap = add space between boxes
        - negative gap = reduce space (overlap)

        Returns:
            List of [x, y, z] positions for each box slot
        """
        positions = []

        # Calculate spacing between box centers
        x_spacing = (self.box_size[0] + self.x_gap) * self.x_direction
        y_spacing = (self.box_size[1] + self.y_gap) * self.y_direction
        z_spacing = (self.box_size[2] + self.z_gap) * self.z_direction

        for z in range(self.z_count):
            for y in range(self.y_count):
                for x in range(self.x_count):
                    position = [
                        self.starting_position[0] + x * x_spacing,
                        self.starting_position[1] + y * y_spacing,
                        self.starting_position[2] + z * z_spacing
                    ]
                    positions.append(position)

        return positions


@dataclass
class BoxState:
    """Current state of box being processed."""

    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    is_waiting: bool = False
    pose_fetched: bool = False


class UR5BoxPicking(Node):
    """ROS2 node for coordinating UR5 box picking and palletizing.

    This node acts as a bridge between the simulation (Isaac Sim) and
    the UR5 palletizing action server, managing the flow of boxes from
    the conveyor to the target pile.

    The node continuously monitors the target box position and velocity,
    controls the conveyor belt, and determines when a box is ready to pick.

    Service Clients:
        /get_prim_attribute: Fetch prim attributes (GetPrimAttribute)
        /set_prim_attribute: Set prim attributes (SetPrimAttribute)
        /box_placed_service: Notify simulation of placed box (Trigger)

    Action Clients:
        ur5/palletizing: Send pick-and-place goals (Palletizing)
    """

    # Node configuration constants
    PROCESS_RATE_HZ = 100.0
    ACTION_SERVER_TIMEOUT_SEC = 10.0
    SERVICE_TIMEOUT_SEC = 1.0

    # Isaac Sim prim paths
    TARGET_BOX_PRIM_PATH = '/World/Robots/objects/boxes/target_box'
    CONVEYOR_PRIM_PATH = '/World/Graphs/SceneCommand/ConveyorBelt'

    # Conveyor control thresholds
    POSITION_THRESHOLD = -0.3  # Y position threshold for stopping conveyor
    VELOCITY_THRESHOLD = 0.01  # Velocity magnitude threshold for box_waiting
    CONVEYOR_VELOCITY_ON = 0.4
    CONVEYOR_VELOCITY_OFF = 0.0

    def __init__(self) -> None:
        """Initialize the UR5 box picking node."""
        super().__init__('ur5_box_picking')

        # Use reentrant callback group for async operations
        self._callback_group = ReentrantCallbackGroup()

        # Load and validate parameters
        self._pile_config = self._load_parameters()

        # Generate target positions
        self._target_positions = self._pile_config.generate_positions()
        self._log_target_positions()

        # Initialize state
        self._box_state = BoxState()
        self._current_pile_index = 0
        self._boxes_placed_count = 0
        self._last_reported_count = 0
        self._is_goal_in_progress = False
        self._is_fetching_pose = False

        # Setup ROS2 interfaces
        self._setup_action_client()
        self._setup_service_clients()

        # Main processing timer
        timer_period = 1.0 / self.PROCESS_RATE_HZ
        self._timer = self.create_timer(
            timer_period,
            self._process_callback,
            callback_group=self._callback_group
        )

        # Initialize conveyor with velocity on (one-shot timer to wait for service)
        self._init_timer = self.create_timer(
            1.0,  # 1 second delay for service to be ready
            self._initialize_conveyor,
            callback_group=self._callback_group
        )

        self.get_logger().info('Node initialized successfully')

    def _initialize_conveyor(self) -> None:
        """Initialize conveyor velocity at startup."""
        self._init_timer.cancel()  # One-shot timer
        self._set_conveyor_velocity(self.CONVEYOR_VELOCITY_ON)
        self.get_logger().info(
            f'Conveyor initialized with velocity {self.CONVEYOR_VELOCITY_ON}'
        )

    def _load_parameters(self) -> PileConfig:
        """Load and validate ROS parameters.

        Returns:
            PileConfig with loaded parameters

        Raises:
            RuntimeError: If required parameters are missing
        """
        _, declared_parameters = get_ros_parameters("ur5_palletizing")
        self.declare_parameters(namespace='', parameters=declared_parameters)

        self.get_logger().info('Loaded parameters:')
        for param, value in declared_parameters:
            self.get_logger().debug(f'  {param}: {value}')

        try:
            config = PileConfig(
                starting_position=list(
                    self.get_parameter("starting_pile_position").value
                ),
                box_size=list(self.get_parameter("box_size").value),
                x_gap=self.get_parameter("x_gap").value,
                y_gap=self.get_parameter("y_gap").value,
                z_gap=self.get_parameter("z_gap").value,
                x_direction=self.get_parameter("x_direction").value,
                y_direction=self.get_parameter("y_direction").value,
                z_direction=self.get_parameter("z_direction").value,
                x_count=self.get_parameter("box_x_count").value,
                y_count=self.get_parameter("box_y_count").value,
                z_count=self.get_parameter("box_z_count").value,
            )
        except rclpy.exceptions.ParameterUninitializedException as e:
            self.get_logger().fatal(f'Missing required parameter: {e}')
            raise RuntimeError(f'Missing required parameter: {e}') from e

        self.get_logger().info(
            f'Pile configuration: {config.x_count}x{config.y_count}x{config.z_count} '
            f'= {config.max_boxes} boxes'
        )

        return config

    def _log_target_positions(self) -> None:
        """Log all target pile positions."""
        self.get_logger().info('Target pile positions:')
        for idx, position in enumerate(self._target_positions):
            self.get_logger().info(f'  [{idx}]: {position}')

    def _setup_action_client(self) -> None:
        """Setup the palletizing action client."""
        self._action_client = ActionClient(
            self,
            Palletizing,
            'ur5/palletizing',
            callback_group=self._callback_group
        )

    def _setup_service_clients(self) -> None:
        """Setup ROS2 service clients."""
        self._box_placed_client = self.create_client(
            Trigger,
            '/box_placed_service',
            callback_group=self._callback_group
        )
        self._get_prim_attr_client = self.create_client(
            GetPrimAttribute,
            '/get_prim_attribute',
            callback_group=self._callback_group
        )
        self._set_prim_attr_client = self.create_client(
            SetPrimAttribute,
            '/set_prim_attribute',
            callback_group=self._callback_group
        )

    # =========================================================================
    # Prim Attribute Service Methods
    # =========================================================================

    def _fetch_box_pose(self) -> None:
        """Fetch box translation and orientation from Isaac Sim."""
        if not self._get_prim_attr_client.service_is_ready():
            self.get_logger().warn(
                'get_prim_attribute service not available',
                throttle_duration_sec=5.0
            )
            return

        self._is_fetching_pose = True

        # Fetch translation
        self._fetch_translation()

    def _fetch_translation(self) -> None:
        """Fetch box translation from Isaac Sim."""
        request = GetPrimAttribute.Request()
        request.path = self.TARGET_BOX_PRIM_PATH
        request.attribute = 'xformOp:translate'

        future = self._get_prim_attr_client.call_async(request)
        future.add_done_callback(self._on_translation_response)

    def _on_translation_response(self, future) -> None:
        """Handle translation service response."""
        try:
            response = future.result()
            if response.success:
                # Parse JSON array string to list
                position = json.loads(response.value)
                self._box_state.position = list(position)
                self.get_logger().debug(f'Box position: {self._box_state.position}')

                # Now fetch velocity
                self._fetch_velocity()
            else:
                self.get_logger().warn(f'Failed to get translation: {response.message}')
                self._is_fetching_pose = False
        except Exception as e:
            self.get_logger().error(f'Translation service call failed: {e}')
            self._is_fetching_pose = False

    def _fetch_velocity(self) -> None:
        """Fetch box velocity from Isaac Sim."""
        request = GetPrimAttribute.Request()
        request.path = self.TARGET_BOX_PRIM_PATH
        request.attribute = 'physics:velocity'

        future = self._get_prim_attr_client.call_async(request)
        future.add_done_callback(self._on_velocity_response)

    def _on_velocity_response(self, future) -> None:
        """Handle velocity service response."""
        try:
            response = future.result()
            if response.success:
                # Parse JSON array string to list [vx, vy, vz]
                velocity = json.loads(response.value)
                self._box_state.velocity = list(velocity)
                self.get_logger().debug(f'Box velocity: {self._box_state.velocity}')

                # Now fetch orientation
                self._fetch_orientation()
            else:
                self.get_logger().warn(f'Failed to get velocity: {response.message}')
                self._is_fetching_pose = False
        except Exception as e:
            self.get_logger().error(f'Velocity service call failed: {e}')
            self._is_fetching_pose = False

    def _fetch_orientation(self) -> None:
        """Fetch box orientation from Isaac Sim."""
        request = GetPrimAttribute.Request()
        request.path = self.TARGET_BOX_PRIM_PATH
        request.attribute = 'xformOp:orient'

        future = self._get_prim_attr_client.call_async(request)
        future.add_done_callback(self._on_orientation_response)

    def _on_orientation_response(self, future) -> None:
        """Handle orientation service response."""
        try:
            response = future.result()
            if response.success:
                # Parse JSON array string to list
                # Isaac Sim returns quaternion as [w, x, y, z]
                # ROS uses [x, y, z, w], so we need to convert
                quat_wxyz = json.loads(response.value)
                # Convert from [w, x, y, z] to [x, y, z, w]
                self._box_state.orientation = [
                    quat_wxyz[1],  # x
                    quat_wxyz[2],  # y
                    quat_wxyz[3],  # z
                    quat_wxyz[0],  # w
                ]
                self.get_logger().debug(
                    f'Box orientation (xyzw): {self._box_state.orientation}'
                )
                self._box_state.pose_fetched = True

                # Update conveyor control after fetch completes
                self._update_conveyor_and_waiting_state()

                # Send goal immediately if box is ready
                if self._should_send_goal():
                    self._send_palletizing_goal()
            else:
                self.get_logger().warn(f'Failed to get orientation: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Orientation service call failed: {e}')
        finally:
            self._is_fetching_pose = False

    # =========================================================================
    # Action Client Methods
    # =========================================================================

    def _send_palletizing_goal(self) -> None:
        """Send a palletizing goal to the action server."""
        if not self._action_client.server_is_ready():
            self.get_logger().warn(
                'Action server not ready, waiting...',
                throttle_duration_sec=5.0
            )
            return

        goal = self._build_goal_message()
        self._is_goal_in_progress = True

        self.get_logger().info(
            f'Sending goal for box {self._current_pile_index + 1}/'
            f'{self._pile_config.max_boxes}'
        )
        self.get_logger().info(
            f'  Start position: {self._box_state.position}'
        )
        self.get_logger().info(
            f'  Target position: {self._target_positions[self._current_pile_index]}'
        )

        future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self._on_goal_feedback
        )
        future.add_done_callback(self._on_goal_response)

    def _build_goal_message(self) -> Palletizing.Goal:
        """Build the palletizing goal message.

        Returns:
            Configured Palletizing.Goal message
        """
        goal = Palletizing.Goal()

        # Source pose (current box position)
        goal.box_start_pose.position.x = self._box_state.position[0]
        goal.box_start_pose.position.y = self._box_state.position[1]
        goal.box_start_pose.position.z = self._box_state.position[2]
        goal.box_start_pose.orientation.x = float(self._box_state.orientation[0])
        goal.box_start_pose.orientation.y = float(self._box_state.orientation[1])
        goal.box_start_pose.orientation.z = float(self._box_state.orientation[2])
        goal.box_start_pose.orientation.w = float(self._box_state.orientation[3])

        # Target pose (pile position)
        target = self._target_positions[self._current_pile_index]
        goal.box_target_pose.position.x = target[0]
        goal.box_target_pose.position.y = target[1]
        goal.box_target_pose.position.z = target[2]

        return goal

    def _on_goal_feedback(self, feedback_msg) -> None:
        """Handle action feedback."""
        self.get_logger().debug(
            f'Goal feedback: {feedback_msg.feedback}',
            throttle_duration_sec=1.0
        )

    def _on_goal_response(self, future) -> None:
        """Handle goal acceptance/rejection response."""
        goal_handle: Optional[ClientGoalHandle] = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by action server')
            self._is_goal_in_progress = False
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        """Handle goal completion result."""
        try:
            result = future.result().result

            if result.error_code == Palletizing.Result.SUCCESS:
                self._current_pile_index += 1
                self._boxes_placed_count += 1
                self.get_logger().info(
                    f'Goal succeeded! Boxes placed: {self._boxes_placed_count}/'
                    f'{self._pile_config.max_boxes}'
                )

                # Reset state for next box - wait for new box to arrive and stop
                self._box_state.is_waiting = False
                self._box_state.pose_fetched = False
            else:
                self.get_logger().error(
                    f'Goal failed with error: {result.error_string}'
                )
        except Exception as e:
            self.get_logger().error(f'Error getting goal result: {e}')
        finally:
            self._is_goal_in_progress = False

    # =========================================================================
    # Service Client Methods
    # =========================================================================

    def _notify_box_placed(self) -> None:
        """Notify simulation that a box was placed."""
        if not self._box_placed_client.service_is_ready():
            self.get_logger().warn(
                'box_placed_service not available',
                throttle_duration_sec=5.0
            )
            return

        request = Trigger.Request()
        future = self._box_placed_client.call_async(request)
        future.add_done_callback(self._on_box_placed_response)

    def _on_box_placed_response(self, future) -> None:
        """Handle box_placed service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().debug(f'Simulation acknowledged: {response.message}')
            else:
                self.get_logger().warn(f'Simulation error: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    # =========================================================================
    # Conveyor Control Methods
    # =========================================================================

    def _update_conveyor_and_waiting_state(self) -> None:
        """Update conveyor velocity and box_waiting state based on box pose."""
        y_position = self._box_state.position[1]  # Y axis
        velocity_magnitude = sqrt(sum(v ** 2 for v in self._box_state.velocity))

        # Control conveyor velocity based on Y position
        if y_position > self.POSITION_THRESHOLD:
            self._set_conveyor_velocity(self.CONVEYOR_VELOCITY_OFF)
        else:
            self._set_conveyor_velocity(self.CONVEYOR_VELOCITY_ON)

        # Determine box_waiting based on position and velocity
        was_waiting = self._box_state.is_waiting
        self._box_state.is_waiting = (
            y_position > self.POSITION_THRESHOLD and
            velocity_magnitude < self.VELOCITY_THRESHOLD
        )

        # Log when box becomes ready for picking
        if self._box_state.is_waiting and not was_waiting:
            self.get_logger().info(
                f'Box ready for picking at Y={y_position:.3f}, '
                f'velocity={velocity_magnitude:.4f}'
            )

    def _set_conveyor_velocity(self, velocity: float) -> None:
        """Set the conveyor belt velocity.

        Args:
            velocity: Target velocity for the conveyor belt
        """
        if not self._set_prim_attr_client.service_is_ready():
            return

        request = SetPrimAttribute.Request()
        request.path = self.CONVEYOR_PRIM_PATH
        request.attribute = 'inputs:velocity'
        request.value = str(velocity)
        self._set_prim_attr_client.call_async(request)

    # =========================================================================
    # Main Processing Loop
    # =========================================================================

    def _process_callback(self) -> None:
        """Main processing loop executed at PROCESS_RATE_HZ."""
        # Notify simulation when a box is placed
        if self._boxes_placed_count > self._last_reported_count:
            self._notify_box_placed()
            self.get_logger().info(
                f'Box placed notification sent (count: {self._boxes_placed_count})'
            )
            self._last_reported_count = self._boxes_placed_count

        # Continuously fetch box state for monitoring (when not busy)
        # Conveyor control is updated in _on_orientation_response callback
        if self._should_fetch_state():
            self._fetch_box_pose()
            return

        # Check if we should send a new goal
        if self._should_send_goal():
            self._send_palletizing_goal()

    def _should_fetch_state(self) -> bool:
        """Determine if we should fetch the box state for monitoring.

        Returns:
            True if state should be fetched
        """
        if self._is_fetching_pose:
            return False

        if self._is_goal_in_progress:
            return False

        return True

    def _should_send_goal(self) -> bool:
        """Determine if conditions are met to send a new goal.

        Returns:
            True if a new goal should be sent
        """
        if self._is_goal_in_progress:
            return False

        if not self._box_state.is_waiting:
            return False

        if not self._box_state.pose_fetched:
            return False

        if self._current_pile_index >= self._pile_config.max_boxes:
            self.get_logger().info(
                'All boxes placed!',
                throttle_duration_sec=10.0
            )
            return False

        return True


def main(args=None) -> None:
    """Entry point for the UR5 box picking node."""
    rclpy.init(args=args)

    node = UR5BoxPicking()

    try:
        node.get_logger().info('Starting node, shut down with CTRL-C')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    except Exception as e:
        node.get_logger().fatal(f'Unexpected error: {e}')
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
