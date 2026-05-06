"""
survey.launch.py — camera + cv + triangulation only.

Typical use: drive gantry to survey position manually, then trigger a burst
via:  ros2 topic pub /lw/cv/trigger_burst std_msgs/msg/Int32 '{data: 5}' --once

Results appear on /lw/targets.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    left_cam_arg  = DeclareLaunchArgument("left_cam",  default_value="0")
    right_cam_arg = DeclareLaunchArgument("right_cam", default_value="2")

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

    cv = Node(
        package="laser_weeder",
        executable="cv_node",
        name="cv_node",
        output="screen",
        parameters=[{"live_hz": 5.0}],
    )

    triangulation = Node(
        package="laser_weeder",
        executable="triangulation_node",
        name="triangulation_node",
        output="screen",
    )

    return LaunchDescription([
        left_cam_arg,
        right_cam_arg,
        camera,
        cv,
        triangulation,
    ])
