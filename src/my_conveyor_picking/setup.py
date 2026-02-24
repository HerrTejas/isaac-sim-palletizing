"""Setup file for my_conveyor_picking package."""

import os
from setuptools import setup

PACKAGE_NAME = "my_conveyor_picking"
PACKAGES_LIST = [
    PACKAGE_NAME,
    "my_conveyor_picking.helper_functions",
]

DATA_FILES = [
    ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
    ("share/" + PACKAGE_NAME, ["package.xml"]),
]


def package_files(data_files, directory_list):
    """Get all files in directories and return list of tuples for installation."""
    paths_dict = {}
    for directory in directory_list:
        if not os.path.exists(directory):
            continue
        for (path, _, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                install_path = os.path.join("share", PACKAGE_NAME, path)

                if install_path in paths_dict:
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]

    for key, value in paths_dict.items():
        data_files.append((key, value))

    return data_files


setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=PACKAGES_LIST,
    data_files=package_files(DATA_FILES, ["config/", "launch/", "usd/"]),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="herrtejas",
    maintainer_email="herrtejasmurkute@gmail.com",
    description="Keyboard-based conveyor belt control for Isaac Sim",
    license="BSD-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "conveyor_keyboard_control = my_conveyor_picking.conveyor_keyboard_control:main",
            "lightbeam_conveyor_stop = my_conveyor_picking.lightbeam_conveyor_stop:main",
            "palletizing_server = my_conveyor_picking.palletizing_server:main",
            "ur5_controller_server = my_conveyor_picking.ur5_controller_server:main",
        ],
    },
)