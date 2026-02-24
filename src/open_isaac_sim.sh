#!/bin/bash

# Script to open Isaac Sim GUI
# Isaac Sim 5.1.0 installed via: pip install isaacsim[all]==5.1.0
#
# FOR UBUNTU 24.04 + ROS2 JAZZY
# NOTE: Isaac Sim 5.1.0 REQUIRES Python 3.11 (not 3.12!)
#       But we use the Jazzy ROS2 bridge for communication

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================
# Virtual Environment Configuration
# ============================================================
# Isaac Sim MUST be installed in a Python 3.11 virtual environment
ISAACSIM_VENV="$HOME/isaacsim_env"

# Check if the virtual environment exists
if [ ! -d "$ISAACSIM_VENV" ]; then
    echo "ERROR: Isaac Sim virtual environment not found at $ISAACSIM_VENV"
    echo ""
    echo "Please create the virtual environment and install Isaac Sim:"
    echo "  python3.11 -m venv $ISAACSIM_VENV"
    echo "  source $ISAACSIM_VENV/bin/activate"
    echo "  pip install --upgrade pip"
    echo "  pip install isaacsim[all]==5.1.0 --extra-index-url https://pypi.nvidia.com"
    exit 1
fi

# Activate the virtual environment
echo "Activating Isaac Sim virtual environment..."
source "$ISAACSIM_VENV/bin/activate"

echo "Opening Isaac Sim 5.1.0 GUI..."
echo ""

# ============================================================
# ROS2 Configuration - Use Isaac Sim's bundled ROS2 Jazzy
# ============================================================
# Isaac Sim 5.1.0 includes ROS2 Jazzy libraries
# Using Jazzy for Ubuntu 24.04 compatibility

# Unset system ROS2 environment variables to avoid conflicts
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH

# Use Python 3.11 (Isaac Sim requirement)
PYTHON_CMD="python3.11"

# Isaac Sim ROS2 Jazzy paths (inside the virtual environment)
# Note: Python 3.11 venv, but using Jazzy ROS2 bridge
ISAAC_ROS2_BASE="$ISAACSIM_VENV/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy"
ISAAC_ROS2_PYTHON="$ISAAC_ROS2_BASE/rclpy"
ISAAC_ROS2_LIB="$ISAAC_ROS2_BASE/lib"

# Set ROS2 Jazzy environment
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3.11
export ROS_DISTRO=jazzy

# Add Isaac Sim's ROS2 Python packages to PYTHONPATH (before system paths)
export PYTHONPATH="$ISAAC_ROS2_PYTHON:$SCRIPT_DIR:$PYTHONPATH"

# Add Isaac Sim's ROS2 shared libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$ISAAC_ROS2_LIB:$LD_LIBRARY_PATH"

# Accept NVIDIA Omniverse EULA automatically
export OMNI_KIT_ACCEPT_EULA=YES

# Set RMW implementation for Jazzy (Cyclone DDS)
# Make sure to use the same RMW in your ROS2 terminals!
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Fix for CUDA LIDAR PTX compilation error (Isaac Sim 5.1.0 known issue)
# Add Isaac Sim sensor CUDA libraries to LD_LIBRARY_PATH
SENSOR_COMMON_BIN="$HOME/.local/share/ov/data/exts/v2/omni.sensors.nv.common-2.5.0-coreapi+lx64.r.cp311/bin"
SENSOR_LIDAR_BIN="$HOME/.local/share/ov/data/exts/v2/omni.sensors.nv.lidar-2.6.3-coreapi+lx64.r.cp311/bin"

if [ -d "$SENSOR_COMMON_BIN" ] && [ -d "$SENSOR_LIDAR_BIN" ]; then
    export LD_LIBRARY_PATH="$SENSOR_LIDAR_BIN:$SENSOR_COMMON_BIN:$LD_LIBRARY_PATH"
fi

# Launch Isaac Sim GUI or run a Python script
if [[ "$1" == *.py ]]; then
    echo "Running Python script with $PYTHON_CMD: $1"
    $PYTHON_CMD "$@"
elif [[ "$1" == "python" || "$1" == "python3" || "$1" == "python3.11" ]]; then
    shift
    echo "Running Python script with $PYTHON_CMD: $1"
    $PYTHON_CMD "$@"
else
    echo "============================================"
    echo "REMINDER: In your ROS2 terminals, run:"
    echo "  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
    echo "============================================"
    isaacsim "$@"
fi
