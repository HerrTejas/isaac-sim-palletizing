import os
from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml_file(filename) -> dict:
    """Load yaml file with the UR5 Isaac Sim Parameters"""
    with open(filename, 'r', encoding='UTF-8') as file:
        data = yaml.safe_load(file)
    return data


def flatten_params(params: dict, prefix: str = "") -> list:
    """Flatten nested parameters into dot-notation keys.

    Parameters
    ----------
    params : dict
        Dictionary of parameters, possibly nested.
    prefix : str
        Prefix for nested keys (used in recursion).

    Returns
    -------
    list
        List of tuples (key, value) with flattened keys.
    """
    result = []
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            # Recursively flatten nested dicts
            result.extend(flatten_params(value, full_key))
        else:
            result.append((full_key, value))
    return result


def get_ros_parameters(node_name):
    """Get the ROS2 parameters from the yaml file.

    Parameters
    ----------
    node_name : str
        Name of the node to get parameters for.

    Returns
    -------
    dict
        ROS2 parameters (original nested structure).
    list
        Declared parameters as flat list of (key, value) tuples.
        Nested parameters are flattened with dot notation
        (e.g., 'end_effector_offset.position').

    """
    # Get the parameters from the yaml file
    config_file = os.path.join(
        get_package_share_directory("my_conveyor_picking"),
        'config',
        'params.yaml'
    )
    config = load_yaml_file(config_file)
    ros_parameters = config[node_name]["ros__parameters"]

    # Declare the parameters in the ROS2 parameter server
    # Flatten nested dicts to support dot-notation access
    declared_parameters = flatten_params(ros_parameters)
    return ros_parameters, declared_parameters
