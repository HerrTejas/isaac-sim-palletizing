"""Launch file for conveyor keyboard control."""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for conveyor control."""
    pkg_dir = get_package_share_directory("my_conveyor_picking")
    config_file = os.path.join(pkg_dir, "config", "conveyor_params.yaml")

    conveyor_control_node = Node(
        package="my_conveyor_picking",
        executable="conveyor_keyboard_control",
        name="conveyor_keyboard_control",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([conveyor_control_node])
