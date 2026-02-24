"""Launch file for palletizing system.

Starts the UR5 controller, palletizing server, and lightbeam conveyor stop nodes.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for palletizing system."""
    pkg_dir = get_package_share_directory("my_conveyor_picking")
    params_file = os.path.join(pkg_dir, "config", "params.yaml")
    conveyor_params_file = os.path.join(pkg_dir, "config", "conveyor_params.yaml")

    ur5_controller_server = Node(
        package="my_conveyor_picking",
        executable="ur5_controller_server",
        name="ur5_controller_server",
        output="screen",
        parameters=[params_file],
    )

    palletizing_server = Node(
        package="my_conveyor_picking",
        executable="palletizing_server",
        name="palletizing_server",
        output="screen",
        parameters=[params_file],
    )

    lightbeam_conveyor_stop = Node(
        package="my_conveyor_picking",
        executable="lightbeam_conveyor_stop",
        name="lightbeam_conveyor_stop",
        output="screen",
        parameters=[conveyor_params_file],
    )

    return LaunchDescription([
        ur5_controller_server,
        palletizing_server,
        lightbeam_conveyor_stop,
    ])
