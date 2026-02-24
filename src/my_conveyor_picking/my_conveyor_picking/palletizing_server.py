"""Palletizing Action Server.
Receives box pose and moves the UR5 robot arm above the box.
"""
from __future__ import annotations
import time
from typing import TYPE_CHECKING
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Bool
from rclpy.action import ActionClient, ActionServer
from tf_transformations import euler_from_quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint
from palletizing_interfaces.action import Palletizing
from my_conveyor_picking.helper_functions import transformations
from my_conveyor_picking.helper_functions.inv_kin import InverseKinematics
from my_conveyor_picking.helper_functions.load_ros_parameters import (
    get_ros_parameters,
)
from my_conveyor_picking.helper_functions.pile_calculator import PileCalculator
if TYPE_CHECKING:
    from rclpy.action.server import ServerGoalHandle
class PalletizingServer(Node):
    """Action server that receives box poses and moves robot above box."""
    def __init__(self) -> None:
        """Initialize the palletizing server."""
        super().__init__("palletizing_server")
        ###############################
        # ROS PARAMETERS
        ###############################
        _, declared_parameters = get_ros_parameters("palletizing_server")
        self.declare_parameters(namespace='', parameters=declared_parameters)
        self.get_logger().info("Parameters:")
        for param, value in declared_parameters:
            self.get_logger().info(f"\t[Palletizing] {param}: {value}")
        self.joint_names = self.get_parameter("joint_names").value
        self.trajectory_time_slow = self.get_parameter(
            "trajectory_time_slow").value
        self.trajectory_time_fast = self.get_parameter(
            "trajectory_time_fast").value
        self.safe_approach_height = self.get_parameter(
            "safe_approach_height").value
        self.box_size = self.get_parameter("box_size").value
        self.gripper_contact_height = self.get_parameter(
            "gripper_contact_height").value
        self.place_release_offset = self.get_parameter(
            "place_release_offset").value
        self.ee_offset_position = self.get_parameter(
            "end_effector_offset.position").value
        self.ee_offset_orientation = self.get_parameter(
            "end_effector_offset.orientation").value
        # Pile configuration
        starting_pile_position = self.get_parameter(
            "starting_pile_position").value
        x_gap = self.get_parameter("x_gap").value
        y_gap = self.get_parameter("y_gap").value
        z_gap = self.get_parameter("z_gap").value
        x_direction = self.get_parameter("x_direction").value
        y_direction = self.get_parameter("y_direction").value
        z_direction = self.get_parameter("z_direction").value
        box_x_count = self.get_parameter("box_x_count").value
        box_y_count = self.get_parameter("box_y_count").value
        box_z_count = self.get_parameter("box_z_count").value
        ###############################
        # PILE CALCULATOR
        ###############################
        self._pile_calculator = PileCalculator(
            starting_position=starting_pile_position,
            box_size=self.box_size,
            x_gap=x_gap,
            y_gap=y_gap,
            z_gap=z_gap,
            x_direction=x_direction,
            y_direction=y_direction,
            z_direction=z_direction,
            box_x_count=box_x_count,
            box_y_count=box_y_count,
            box_z_count=box_z_count
        )
        self.get_logger().info(
            f"Pile capacity: {self._pile_calculator.total_capacity} boxes"
        )
        ###############################
        # INVERSE KINEMATICS
        ###############################
        self.ik_solver = InverseKinematics(
            solution_index=5,
            ee_offset_position=self.ee_offset_position,
            ee_offset_orientation=self.ee_offset_orientation
        )
        ###############################
        # ACTION CLIENT (to UR5 controller)
        ###############################
        client_cb_group = ReentrantCallbackGroup()
        self.ur5_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            'ur5/follow_joint_trajectory',
            callback_group=client_cb_group
        )
        ###############################
        # ACTION SERVER
        ###############################
        self._action_server = ActionServer(
            self,
            Palletizing,
            "palletizing",
            self._execute_callback,
            callback_group=client_cb_group
        )
        ###############################
        # GRIPPER PUBLISHER
        ###############################
        self._gripper_pub = self.create_publisher(
            Bool,
            'gripper_closed',
            10
        )
        ###############################
        # STATE TRACKING
        ###############################
        self._current_step = 1
        self._prev_step = 0
        self._total_steps = 7
        self._box_moving = False
        self._target_pile_position: list[float] = []
        self.get_logger().info("Palletizing Server initialized")
        self.get_logger().info("Waiting for palletizing goals...")
    def _toggle_gripper(self, wait_time: float = 0.5) -> None:
        """Toggle gripper state by pulsing ON then OFF.
        The gripper in Isaac Sim responds to signal transitions.
        This method sends True then False to trigger the gripper action.
        Args:
            wait_time: Time to wait after toggling for action to complete.
        """
        self._gripper_pub.publish(Bool(data=True))
        time.sleep(0.1)  # Brief pulse
        self._gripper_pub.publish(Bool(data=False))
        time.sleep(wait_time)  # Wait for gripper action to complete
    def _compute_place_orientation(
        self,
        box_orientation: list[float]
    ) -> list[float]:
        """Compute gripper orientation for placing box with correct alignment.
        When placing, the gripper orientation must compensate for the box's
        current rotation to align the box to 90 degrees (or -90 due to symmetry)
        using the shortest rotation path.
        Args:
            box_orientation: Box quaternion [x, y, z, w].
        Returns:
            Gripper orientation [roll, pitch, yaw] in degrees.
        """
        # Convert quaternion to euler angles
        euler_angles = euler_from_quaternion(box_orientation)
        z_angle = np.degrees(euler_angles[2])
        # Compute shortest angle difference to align box to 90 degrees
        # Consider both +90 and -90 (equivalent due to box symmetry)
        # Normalize to [-180, 180] range for shortest path
        z_diff_to_90 = ((90.0 - z_angle) + 180.0) % 360.0 - 180.0
        z_diff_to_neg90 = ((-90.0 - z_angle) + 180.0) % 360.0 - 180.0
        # Choose the direction with minimum rotation
        if abs(z_diff_to_90) <= abs(z_diff_to_neg90):
            z_diff = z_diff_to_90
        else:
            z_diff = z_diff_to_neg90
        self.get_logger().info(
            f"Box Z angle: {z_angle:.2f}, correction: {z_diff:.2f}"
        )
        return [180.0, 0.0, 180.0 + z_diff]
    async def _execute_callback(
        self,
        goal_handle: ServerGoalHandle
    ) -> Palletizing.Result:
        """Handle incoming palletizing goal.
        Args:
            goal_handle: The goal handle containing box poses.
        Returns:
            Palletizing.Result with operation status.
        """
        self.get_logger().info("Received palletizing goal!")
        self.ik_solver.reset()
        # Reset state for new palletizing sequence
        self._current_step = 1
        self._prev_step = 0
        # Check if already moving a box
        if self._box_moving:
            result = Palletizing.Result()
            result.error_code = result.FAILED
            result.error_string = "A box is already being moved"
            self.get_logger().warn(result.error_string)
            return result
        # Extract box start pose
        start_pose = goal_handle.request.box_start_pose
        self.get_logger().info("Box Start Pose:")
        self.get_logger().info(
            f"  Position: x={start_pose.position.x:.3f}, "
            f"y={start_pose.position.y:.3f}, z={start_pose.position.z:.3f}"
        )
        # Get target pile position
        self._target_pile_position = self._pile_calculator.get_next_position()
        if self._target_pile_position is None:
            result = Palletizing.Result()
            result.error_code = result.FAILED
            result.error_string = "Pile is full, cannot place more boxes"
            self.get_logger().error(result.error_string)
            goal_handle.abort()
            return result
        self.get_logger().info(
            f"Target pile position: {self._target_pile_position}"
        )
        self.get_logger().info(
            f"Boxes placed: {self._pile_calculator.boxes_placed}/"
            f"{self._pile_calculator.total_capacity}"
        )
        # Execute step-based movements
        while self._current_step <= self._total_steps:
            if self._prev_step != self._current_step:
                success = await self._execute_movement(goal_handle)
                if not success:
                    result = Palletizing.Result()
                    result.error_code = result.FAILED
                    result.error_string = f"Failed at step {self._current_step}"
                    goal_handle.abort()
                    return result
        # All steps completed successfully
        result = Palletizing.Result()
        if self._current_step > self._total_steps:
            goal_handle.succeed()
            result.error_code = result.SUCCESS
            result.error_string = "Palletizing sequence completed successfully"
        else:
            goal_handle.abort()
            result.error_code = result.FAILED
            result.error_string = "Palletizing sequence incomplete"
        self.get_logger().info(f"Result: {result.error_string}")
        return result
    async def _execute_movement(
        self,
        goal_handle: ServerGoalHandle
    ) -> bool:
        """Execute the current movement step.
        7-step palletizing sequence:
        1. Move above pick position (slow)
        2. Move down to pick, close gripper (fast)
        3. Lift above pick position (fast)
        4. Move above place position (fast)
        5. Move down to place position (fast)
        6. Open gripper (fast)
        7. Lift above place position (fast)
        Args:
            goal_handle: The goal handle containing box poses.
        Returns:
            True if movement succeeded, False otherwise.
        """
        self._prev_step = self._current_step
        self._box_moving = True
        self.get_logger().info(
            f"Executing step {self._current_step}/{self._total_steps}..."
        )
        # Extract box position from goal
        box_start_pose = goal_handle.request.box_start_pose
        pick_position = [
            box_start_pose.position.x,
            box_start_pose.position.y,
            box_start_pose.position.z
        ]
        box_orientation = [
            box_start_pose.orientation.x,
            box_start_pose.orientation.y,
            box_start_pose.orientation.z,
            box_start_pose.orientation.w
        ]
        # Target position from pile calculator
        place_position = self._target_pile_position.copy()
        # Standard gripper orientation for picking (pointing down)
        pick_gripper_orientation = [180.0, 0.0, 180.0]
        # Corrected orientation for placing (compensates for box rotation)
        place_gripper_orientation = self._compute_place_orientation(box_orientation)
        if self._current_step == 1:
            # Step 1: Move above pick position (slow)
            target = pick_position.copy()
            target[2] += self.safe_approach_height
            self.get_logger().info(f"Step 1: Moving above pick at {target}")
            success = await self._move_to_pose(
                target, pick_gripper_orientation, movement="slow"
            )
        elif self._current_step == 2:
            # Step 2: Move down to pick, close gripper (fast)
            target = pick_position.copy()
            target[2] += self.gripper_contact_height
            self.get_logger().info(f"Step 2: Moving to pick at {target}")
            success = await self._move_to_pose(
                target, pick_gripper_orientation, movement="fast"
            )
            if success:
                self.get_logger().info("Closing gripper...")
                self._toggle_gripper()
                self.get_logger().info("Gripper closed")
        elif self._current_step == 3:
            # Step 3: Lift above pick position (fast)
            target = pick_position.copy()
            target[2] += self.safe_approach_height
            self.get_logger().info(f"Step 3: Lifting from pick at {target}")
            success = await self._move_to_pose(
                target, pick_gripper_orientation, movement="fast"
            )
        elif self._current_step == 4:
            # Step 4: Move above place position (fast)
            target = place_position.copy()
            target[2] += self.safe_approach_height
            self.get_logger().info(f"Step 4: Moving above place at {target}")
            success = await self._move_to_pose(
                target, place_gripper_orientation, movement="fast"
            )
        elif self._current_step == 5:
            # Step 5: Move down to place position (fast)
            target = place_position.copy()
            target[2] += self.gripper_contact_height + self.place_release_offset
            self.get_logger().info(f"Step 5: Moving to place at {target}")
            success = await self._move_to_pose(
                target, place_gripper_orientation, movement="fast"
            )
        elif self._current_step == 6:
            # Step 6: Open gripper - stay at same position
            target = place_position.copy()
            target[2] += self.gripper_contact_height + self.place_release_offset
            self.get_logger().info(f"Step 6: Opening gripper at {target}")
            self._toggle_gripper()
            self.get_logger().info("Box released")
            success = True  # No movement needed
        elif self._current_step == 7:
            # Step 7: Lift above place position (fast)
            target = place_position.copy()
            target[2] += self.safe_approach_height
            self.get_logger().info(f"Step 7: Lifting from place at {target}")
            success = await self._move_to_pose(
                target, place_gripper_orientation, movement="fast"
            )
        # Publish feedback
        feedback = Palletizing.Feedback()
        feedback.end_effector_pose = box_start_pose
        goal_handle.publish_feedback(feedback)
        self._box_moving = False
        if success:
            self._current_step += 1
        return success
    async def _move_to_pose(
        self,
        position: list[float],
        orientation: list[float],
        movement: str = "slow"
    ) -> bool:
        """Move robot end-effector to specified pose.
        Args:
            position: Target [x, y, z] position in meters.
            orientation: Target [roll, pitch, yaw] in degrees.
            movement: Movement speed - "slow" or "fast".
        Returns:
            True if movement succeeded, False otherwise.
        """
        # Build desired gripper pose HTM
        gripper_desired_pose = transformations.get_desired_pose_htm(
            position=np.array(position),
            roll=orientation[0],
            pitch=orientation[1],
            yaw=orientation[2]
        )
        # Solve inverse kinematics
        try:
            target_joint_angles = self.ik_solver.solve(gripper_desired_pose)
        except ValueError as e:
            self.get_logger().error(f"IK solver failed: {e}")
            return False
        self.get_logger().info(
            f"Target joint angles: {np.round(target_joint_angles, 3)}")
        # Select trajectory time based on movement type
        if movement == "fast":
            trajectory_time = self.trajectory_time_fast
        else:
            trajectory_time = self.trajectory_time_slow
        # Build and send trajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        duration = Duration(sec=trajectory_time)
        goal.trajectory.points.append(
            JointTrajectoryPoint(
                positions=target_joint_angles.tolist(),
                velocities=[0.0] * 6,
                accelerations=[0.0] * 6,
                time_from_start=duration
            )
        )
        # Wait for action server
        if not self.ur5_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("UR5 action server not available")
            return False
        # Send goal and wait for result
        self.get_logger().info("Sending trajectory goal to UR5 controller...")
        send_goal_future = await self.ur5_action_client.send_goal_async(goal)
        if not send_goal_future.accepted:
            self.get_logger().error("Goal rejected by UR5 controller")
            return False
        self.get_logger().info("Goal accepted, waiting for result...")
        result_future = await send_goal_future.get_result_async()
        result = result_future.result
        if result.error_string == "Success":
            self.get_logger().info("Robot movement completed successfully")
            return True
        else:
            self.get_logger().error(f"Movement failed: {result.error_string}")
            return False
def main(args: list[str] | None = None) -> None:
    """Main entry point."""
    rclpy.init(args=args)
    node = PalletizingServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        node.get_logger().info(
            "[Palletizing] Starting server, shut down with CTRL-C")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("[Palletizing] Keyboard interrupt, shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == "__main__":
    main()