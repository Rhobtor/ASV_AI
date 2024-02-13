#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --
source "/root/ros2_ws/install/local_setup.bash" --

echo "---"  
echo 'Available ZED packages:'
ros2 pkg list | grep zed
echo "---------------------"    
exec "$@"