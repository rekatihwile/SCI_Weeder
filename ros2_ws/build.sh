#!/usr/bin/env bash
# build.sh — build the LaserWeeder ROS2 workspace.
#
# Usage:
#   cd ros2_ws && ./build.sh              # full build
#   cd ros2_ws && ./build.sh --symlink    # symlink-install (faster dev iteration)
#
# After building, source the install overlay:
#   source install/setup.bash
# Then launch:
#   ros2 launch laser_weeder full_system.launch.py
#   ros2 launch laser_weeder full_system.launch.py mock_gantry:=true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source the ROS2 base install if not already sourced
if [[ -z "${ROS_DISTRO:-}" ]]; then
    # Try common Humble / Iron / Jazzy locations
    for setup in /opt/ros/humble/setup.bash \
                 /opt/ros/iron/setup.bash   \
                 /opt/ros/jazzy/setup.bash; do
        if [[ -f "$setup" ]]; then
            # shellcheck source=/dev/null
            source "$setup"
            echo "[build.sh] Sourced $setup (ROS_DISTRO=${ROS_DISTRO})"
            break
        fi
    done
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "[build.sh] ERROR: ROS2 not found. Source your ROS2 setup.bash first." >&2
    exit 1
fi

SYMLINK_FLAG=""
if [[ "${1:-}" == "--symlink" ]]; then
    SYMLINK_FLAG="--symlink-install"
    echo "[build.sh] Using --symlink-install"
fi

# Build messages first, then the node package
colcon build \
    ${SYMLINK_FLAG} \
    --packages-select laser_weeder_msgs \
    --event-handlers console_direct+

# Source the messages overlay so the node package can find them
# shellcheck source=/dev/null
source install/setup.bash

colcon build \
    ${SYMLINK_FLAG} \
    --packages-select laser_weeder \
    --event-handlers console_direct+

echo ""
echo "[build.sh] Build complete."
echo "Run:  source ${SCRIPT_DIR}/install/setup.bash"
echo "Then: ros2 launch laser_weeder full_system.launch.py"
