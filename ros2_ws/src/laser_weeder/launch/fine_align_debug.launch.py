"""
fine_align_debug.launch.py — camera + gantry + fine_align only.

Typical use: manually publish a FineAlignGoal to test/debug the PD loop
without running CV or triangulation.

Example:
  ros2 topic pub /lw/fine_align/goal laser_weeder_msgs/msg/FineAlignGoal \
    '{left_x_px: 960.0, left_y_px: 540.0, right_x_px: 900.0, right_y_px: 540.0}' \
    --once
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mock_arg      = DeclareLaunchArgument("mock_gantry", default_value="false")
    left_cam_arg  = DeclareLaunchArgument("left_cam",    default_value="0")
    right_cam_arg = DeclareLaunchArgument("right_cam",   default_value="2")

    camera = Node(
        package="laser_weeder",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[{
            "left_index":  LaunchConfiguration("left_cam"),
            "right_index": LaunchConfiguration("right_cam"),
            "fps":         30,
        }],
    )

    gantry = Node(
        package="laser_weeder",
        executable="gantry_node",
        name="gantry_node",
        output="screen",
        parameters=[{
            "mock":      LaunchConfiguration("mock_gantry"),
            "status_hz": 10.0,
        }],
    )

    fine_align = Node(
        package="laser_weeder",
        executable="fine_align_node",
        name="fine_align_node",
        output="screen",
    )

    return LaunchDescription([
        mock_arg,
        left_cam_arg,
        right_cam_arg,
        camera,
        gantry,
        fine_align,
    ])
