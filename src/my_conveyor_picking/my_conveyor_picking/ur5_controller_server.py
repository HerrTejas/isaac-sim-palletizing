#!/usr/bin/env python3
"""UR5 Trajectory Controller Server.

Handles FollowJointTrajectory action for UR5 robot in Isaac Sim.
Supports cubic, quintic, LSPB, and minimum time trajectory interpolation.
"""

from __future__ import annotations

import copy
import math
import time
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from my_conveyor_picking.helper_functions import trajectory_check as tc
from my_conveyor_picking.helper_functions.load_ros_parameters import \
    get_ros_parameters

if TYPE_CHECKING:
    from rclpy.action.server import ServerGoalHandle

# Type alias for trajectory data
TrajectoryData = tuple[list[float], list[float]]


class UR5TrajControllerServer(Node):
    """Action server for UR5 robot trajectory control."""

    def __init__(self) -> None:
        """Initialize the UR5 trajectory controller server."""
        super().__init__('ur5_controller_server')

        ###############################
        # ROS PARAMETERS
        ###############################
        _, declared_parameters =\
            get_ros_parameters("ur5_controller_server")
        self.declare_parameters(namespace='',
                                parameters=declared_parameters)
        self.get_logger().info("Parameters:")
        for param, value in declared_parameters:
            self.get_logger().info(f"\t[UR5 Controller] {param}: {value}")
        self.ur5_joint_names = self.get_parameter(
            "joint_names").value
        self.trajectory_type = self.get_parameter(
            "trajectory_type").value
        self.trajectory_timeout = self.get_parameter(
            "trajectory_timeout").value
        self.trajectory_tolerance_error = self.get_parameter(
            "trajectory_tolerance_error").value

        ###############################
        # PUBLISHER
        ###############################
        self.robot_isaac_pub = self.create_publisher(
            JointState,
            "/joint_command",
            10
        )

        ###############################
        # SUBSCRIBER
        ###############################
        client_cb_group = ReentrantCallbackGroup()
        self.joint_states = None
        self.create_subscription(
            JointState,
            '/joint_states',
            self._update_joint_state,
            10,
            callback_group=client_cb_group
        )

        ###############################
        # ACTION SERVER
        ###############################
        self.robotserver = ActionServer(
            self,
            FollowJointTrajectory,
            "ur5/follow_joint_trajectory",
            self.execute_callback,
            callback_group=client_cb_group
        )
        self.get_logger().info('ur5_controller_server [ON]!')

        self.actual_joint_state = []
        self.trajectory_position_list = []
        self.trajectory_velocity_list = []
        self.trajectory_acceleration_list = []
        self.time_from_start_list = []
        self.time_t0 = 0.0
        self.matrix_a = None
        self.actual_trajectory_to_send = JointState()
        self.received_goal_handle = None
        self.trajectory_in_execution = False
        self.section_width = 30

    def _update_joint_state(self, msg: JointState) -> None:
        """Get simulated UR5 joint angles.

        Joint order: shoulder_pan, shoulder_lift, elbow,
        wrist_1, wrist_2, wrist_3.

        Args:
            msg: JointState message from /joint_states topic.
        """
        self.joint_states = msg.position[0:6]

    async def execute_callback(
        self,
        goal_handle: ServerGoalHandle
    ) -> FollowJointTrajectory.Result:
        """Handle a new goal trajectory command.

        Args:
            goal_handle: ActionServer goal handle.

        Returns:
            FollowJointTrajectory.Result with trajectory execution status.
        """
        if not self.trajectory_in_execution:
            self.get_logger().info('[UR5 Controller] Executing goal...')

            self._init_trajectory_robot()
            initialized_correctly, msg =\
                self._initialize_trajectory(goal_handle)
            if initialized_correctly:
                result, total_time_from_init =\
                    await self._execute_trajectory(goal_handle)
                self.get_logger().info(
                    "[UR5 Controller] Trajectory executed "
                    f"in {total_time_from_init:.2f} seconds"
                )
                goal_handle.succeed()
                return result
            else:
                result = FollowJointTrajectory.Result()
                result._error_code = result.INVALID_GOAL
                result._error_string = msg
                return result

        # set goal_handle as not accepted
        self.get_logger().warn(
            "[UR5 Controller] Trajectory already in execution")
        result = FollowJointTrajectory.Result()
        result._error_code = result.INVALID_GOAL
        result._error_string =\
            "[UR5 Controller] Failed. A trajectory is already in execution"
        return result

    def _init_trajectory_robot(self) -> None:
        """Initialize a new target trajectory."""
        self.actual_joint_state = copy.deepcopy(self.joint_states)

        self.get_logger().info("-" * self.section_width)
        self.get_logger().info(
            f"Actual joint states (UR5): {self.actual_joint_state}")
        self.get_logger().info("-" * self.section_width)

        self.time_t0 = time.time()

    def _initialize_trajectory(
        self,
        received_goal_handle: ServerGoalHandle
    ) -> tuple[bool, str]:
        """Initialize trajectory from goal handle.

        Args:
            received_goal_handle: ActionServer goal handle.

        Returns:
            Tuple of (success: bool, error_message: str).
        """
        self.get_logger().info("[UR5 Controller] Received goal")

        self.received_goal_handle = received_goal_handle
        if not tc.trajectory_is_finite(
                received_goal_handle.request.trajectory):
            error_msg = (
                "[UR5 Controller] Trajectory not executed."
                " Received a goal with infinites or NaNs"
            )
            self.get_logger().error(error_msg)
            received_goal_handle.abort()
            return False, error_msg
        # Checks that the trajectory has velocities
        if not tc.has_velocities(received_goal_handle.request.trajectory):
            error_msg = "[UR5 Controller] Received a goal without velocities"
            self.get_logger().error(error_msg)
            received_goal_handle.abort()
            return False, error_msg

        if self.actual_joint_state is None:
            error_msg = "[UR5 Controller] No joint state received yet"
            self.get_logger().error(error_msg)
            received_goal_handle.abort()
            return False, error_msg

        self.trajectory_position_list = [self.actual_joint_state]
        self.trajectory_velocity_list = [[0.0]*6]
        self.trajectory_acceleration_list = [[0.0]*6]
        self.time_from_start_list = [0.0]
        for point in received_goal_handle.request.trajectory.points:
            self.trajectory_position_list.append(point.positions)
            self.trajectory_velocity_list.append(point.velocities)
            self.trajectory_acceleration_list.append(point.accelerations)
            self.time_from_start_list.append(point.time_from_start.sec)

        if self.trajectory_type == 1:
            self.get_logger().info(
                "[UR5 Controller] Initializing cubic trajectory")
            self._init_interp_cubic()
        elif self.trajectory_type == 2:
            self.get_logger().info(
                "[UR5 Controller] Initializing quintic trajectory")
            self._init_interp_quintic()
        elif self.trajectory_type == 3:
            self.get_logger().info(
                "[UR5 Controller] Initializing LSPB trajectory")
            self._init_lspb_trajectory()
        elif self.trajectory_type == 4:
            self.get_logger().info(
                "[UR5 Controller] Initializing minimum time trajectory")
            self._init_minimum_time_trajectory()

        # Considering that we have only 2 points
        actual_trajectory_to_send =\
            self._sample_trajectory(True)
        self.actual_trajectory_to_send.position = actual_trajectory_to_send[0]
        self.actual_trajectory_to_send.velocity = actual_trajectory_to_send[1]
        return True, ""

    def _init_minimum_time_trajectory(self) -> None:
        """Initialize a minimum time trajectory."""
        self.q0 = self.trajectory_position_list[0]
        self.qf = self.trajectory_position_list[-1]
        self.alfa = [1.0]*6
        self.ts = [0.0]*6
        self.tf = [0.0]*6
        for i in range(6):
            if self.qf[i] - self.q0[i] > 0:
                self.alfa[i] = self.alfa[i]
            else:
                self.alfa[i] = -self.alfa[i]
            self.ts[i] = math.sqrt(
                abs((self.qf[i] - self.q0[i])/self.alfa[i]))
            self.tf[i] = 2*self.ts[i]
        self.time_from_start_list = [max(self.tf)]
        self.pos_atual = [0.0]*6
        self.vel_atual = [0.0]*6

    def _init_lspb_trajectory(self) -> None:
        """Initialize a LSPB trajectory."""
        self.q0 = self.trajectory_position_list[0]
        self.qf = self.trajectory_position_list[-1]
        self.pos_atual = [0.0]*6
        self.vel_atual = [0.0]*6
        self.tf = self.time_from_start_list[-1]
        self.tb = self.tf/3.0
        self.vtb = [0.0]*6
        for i in range(6):
            self.vtb[i] = 1.5*(self.qf[i] - self.q0[i])/self.tf

    def _sample_trajectory(self, first_point: bool = False) -> TrajectoryData:
        """Sample the trajectory at current time.

        Args:
            first_point: If True, return the first point of the trajectory.

        Returns:
            Tuple of (positions, velocities) lists.
        """
        if first_point:
            return self.actual_joint_state, [0.0]*6

        # Last point
        if (time.time() - self.time_t0) >= self.time_from_start_list[-1]:
            return self.trajectory_position_list[-1], [0.0]*6

        if self.trajectory_type == 1:
            trajectory_method = self._update_interp_cubic
        elif self.trajectory_type == 2:
            trajectory_method = self._update_interp_quintic
        elif self.trajectory_type == 3:
            trajectory_method = self._update_lspb_trajectory
        elif self.trajectory_type == 4:
            trajectory_method = self._update_minimum_time_traj
        return trajectory_method(time.time() - self.time_t0)

    def _update_minimum_time_traj(self, t: float) -> TrajectoryData:
        """Update the minimum time trajectory.

        Args:
            t: Time since the trajectory was started.

        Returns:
            Tuple of (positions, velocities) lists.
        """
        for i in range(6):
            if t < self.ts[i]:
                self.pos_atual[i] = self.q0[i] + (self.alfa[i]/2)*t**2
                self.vel_atual[i] = (self.alfa[i])*t
            elif t >= self.ts[i] and t < self.tf[i]:
                self.pos_atual[i] = self.qf[i] \
                    - (self.alfa[i]*self.tf[i]**2)/2 \
                    + self.alfa[i]*self.tf[i]*t \
                    - (self.alfa[i]/2)*t**2
                self.vel_atual[i] = self.alfa[i]*self.tf[i] - self.alfa[i]*t

        return self.pos_atual, self.vel_atual

    def _init_interp_cubic(self) -> None:
        """Initialize cubic polynomial trajectory interpolation."""
        q0 = self.trajectory_position_list[0]
        qf = self.trajectory_position_list[-1]
        v0 = self.trajectory_velocity_list[0]
        vf = self.trajectory_velocity_list[-1]
        t0 = self.time_from_start_list[0]
        tf = self.time_from_start_list[-1]
        a = [0.0]*6
        for i in range(6):
            q0_joint_i = q0[i]
            v0_joint_i = v0[i]
            qf_joint_i = qf[i]
            vf_joint_i = vf[i]
            b = np.array(
                [q0_joint_i, v0_joint_i, qf_joint_i, vf_joint_i]).transpose()
            m = np.array([[1, t0, t0**2,   t0**3],
                          [0,  1,  2*t0, 3*t0**2],
                          [1, tf, tf**2,   tf**3],
                          [0,  1,  2*tf, 3*tf**2]])
            a[i] = np.linalg.inv(m).dot(b)
        self.matrix_a = a

    def _update_interp_cubic(self, t: float) -> TrajectoryData:
        """Update cubic polynomial trajectory.

        Args:
            t: Time since the trajectory was started.

        Returns:
            Tuple of (positions, velocities) lists.
        """
        pos_points, vel_points = [0.0]*6, [0.0]*6
        a = self.matrix_a
        for j in range(6):
            pos_points[j] = a[j][0] + a[j][1]*t + a[j][2]*t**2 + a[j][3]*t**3
            vel_points[j] = a[j][1] + 2*a[j][2]*t + 3*a[j][3]*t**2

        return pos_points, vel_points

    def _init_interp_quintic(self) -> None:
        """Initialize quintic polynomial trajectory interpolation."""
        init_vel = init_acc = end_vel = end_acc = 0
        init_t = self.time_from_start_list[0]
        end_t = self.time_from_start_list[-1]
        joint_angles = [0.0]*6

        for i in range(6):
            init_joint_i = self.trajectory_position_list[0][i]
            end_joint_i = self.trajectory_position_list[-1][i]
            mat_b = np.array([init_joint_i, init_vel, init_acc,
                              end_joint_i, end_vel, end_acc]).transpose()
            mat_m = np.array(
                [[1, init_t, init_t**2, init_t**3, init_t**4, init_t**5],
                 [0, 1, 2*init_t, 3*init_t**2, 4*init_t**3, 5*init_t**4],
                 [0, 0, 2, 6*init_t, 12*init_t**2, 20*init_t**3],
                 [1, end_t, end_t**2, end_t**3, end_t**4, end_t**5],
                 [0, 1, 2*end_t, 3*end_t**2, 4*end_t**3,  5*end_t**4],
                 [0, 0, 2, 6*end_t, 12*end_t**2, 20*end_t**3]])
            joint_angles[i] = np.linalg.inv(mat_m).dot(mat_b)
        self.matrix_a = joint_angles

    def _update_interp_quintic(self, t: float) -> TrajectoryData:
        """Update quintic polynomial trajectory.

        Args:
            t: Time since the trajectory was started.

        Returns:
            Tuple of (positions, velocities) lists.
        """
        pos_points, vel_points, acc_points = [0.0]*6, [0.0]*6, [0.0]*6
        a = self.matrix_a
        for j in range(6):
            pos_points[j] = a[j][0] + a[j][1]*t + a[j][2] * \
                t**2 + a[j][3]*t**3 + a[j][4]*t**4 + a[j][5]*t**5
            vel_points[j] = a[j][1] + 2*a[j][2]*t + 3 * \
                a[j][3]*t**2 + 4*a[j][4]*t**3 + 5*a[j][5]*t**4
            acc_points[j] = 2*a[j][2] + 6*a[j][3] * \
                t + 12*a[j][4]*t**2 + 20*a[j][5]*t**3

        return pos_points, vel_points

    def _update_lspb_trajectory(self, t: float) -> TrajectoryData:
        """Update the LSPB trajectory.

        Args:
            t: Time since the trajectory was started.

        Returns:
            Tuple of (positions, velocities) lists.
        """
        q0 = self.trajectory_position_list[0]
        qf = self.trajectory_position_list[-1]

        for i in range(6):
            alfa = self.vtb[i] / self.tb
            if t <= self.tb:
                self.pos_atual[i] = q0[i] + (alfa/2)*t**2
                self.vel_atual[i] = (alfa)*t
            elif t > self.tb and t <= (self.tf - self.tb):
                self.pos_atual[i] = (
                    qf[i] + q0[i] - self.vtb[i]*self.tf)/2 + self.vtb[i]*t
                self.vel_atual[i] = self.vtb[i]
            elif t > (self.tf - self.tb) and t <= self.tf:
                self.pos_atual[i] = qf[i] - alfa*self.tf**2 / \
                    2 + alfa*self.tf*t - alfa/2*t**2
                self.vel_atual[i] = alfa*self.tf - alfa*t

        return self.pos_atual, self.vel_atual

    async def _execute_trajectory(
        self,
        goal_handle: ServerGoalHandle
    ) -> tuple[FollowJointTrajectory.Result, float]:
        """Execute the trajectory and publish commands to the robot.

        Args:
            goal_handle: ActionServer goal handle.

        Returns:
            Tuple of (FollowJointTrajectory.Result, execution_time).
        """
        self.get_logger().info("[UR5 Controller] Executing trajectory")
        result = FollowJointTrajectory.Result()
        result._error_code = result.INVALID_GOAL
        if goal_handle is not None:
            now = time.time()
            total_time_from_init = now - self.time_t0
            in_time = total_time_from_init <= self.trajectory_timeout
            position_in_tol = tc.within_tolerance(
                self.joint_states,
                self.trajectory_position_list[-1],
                [self.trajectory_tolerance_error] * 6
            )

            self.trajectory_in_execution = True
            while not position_in_tol and in_time:
                position, velocity = self._sample_trajectory()
                self.actual_trajectory_to_send.header.stamp =\
                    self.get_clock().now().to_msg()
                self.actual_trajectory_to_send.name = self.ur5_joint_names
                self.actual_trajectory_to_send.position = position
                self.actual_trajectory_to_send.velocity = velocity

                self.robot_isaac_pub.publish(
                    self.actual_trajectory_to_send)

                now = time.time()
                total_time_from_init = now - self.time_t0
                in_time = total_time_from_init <= self.trajectory_timeout
                position_in_tol = tc.within_tolerance(
                    self.joint_states,
                    self.trajectory_position_list[-1],
                    [self.trajectory_tolerance_error] * 6
                )

            feedback = FollowJointTrajectory.Feedback()
            feedback.header.stamp = self.get_clock().now().to_msg()
            feedback.joint_names = self.ur5_joint_names
            feedback.desired.positions = self.trajectory_position_list[-1]
            feedback.actual.positions = self.joint_states
            feedback.error.positions = [
                a - b for a, b in zip(
                    self.trajectory_position_list[-1], self.joint_states)]
            self.get_logger().info("[UR5 Controller] Feedback:")
            self.get_logger().info(f"\tDesired: {feedback.desired.positions}")
            self.get_logger().info(f"\tActual: {feedback.actual.positions}")
            self.get_logger().info(f"\tError: {feedback.error.positions}")
            self.received_goal_handle.publish_feedback(feedback)

            if position_in_tol:
                result._error_code = result.SUCCESSFUL
                result._error_string = "Success"
                self.received_goal_handle = None
            else:
                result._error_string =\
                    ("[UR5 Controller] Failed."
                     "Joint states not in tolerance. Time exceeded.")
                self.get_logger().warn(
                    f"[UR5 Controller] [TIMEOUT] {total_time_from_init}"
                    "seconds")

            self.trajectory_in_execution = False
            return result, total_time_from_init


def main(args: list[str] | None = None) -> None:
    """Main entry point."""
    rclpy.init(args=args)
    ur5_isaac_server = UR5TrajControllerServer()
    executor = MultiThreadedExecutor()
    executor.add_node(ur5_isaac_server)

    try:
        ur5_isaac_server.get_logger().info(
            '[UR5 Controller] Starting client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        ur5_isaac_server.get_logger().info(
            '[UR5 Controller] Keyboard interrupt, shutting down.\n')
    ur5_isaac_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
