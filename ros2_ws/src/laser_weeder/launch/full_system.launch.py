"""
full_system.launch.py — starts all 6 LaserWeeder nodes.

Nodes:
  camera_node        — capture and publish raw stereo frames
  recorder_node      — save stereo pairs to disk on command
  cv_node            — YOLO detection (live + burst)
  triangulation_node — stereo match + triangulate burst detections
  gantry_node        — gantry serial driver
  fine_align_node    — LK optical-flow PD servo loop

Launch arguments:
  mock_gantry   bool   Use MockGantry (default false)
  left_cam      int    Left camera index (default 0)
  right_cam     int    Right camera index (default 2)
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

    recorder = Node(
        package="laser_weeder",
        executable="recorder_node",
        name="recorder_node",
        output="screen",
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

    gantry = Node(
        package="laser_weeder",
        executable="gantry_node",
        name="gantry_node",
        output="screen",
        parameters=[{
            "mock":       LaunchConfiguration("mock_gantry"),
            "status_hz":  10.0,
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
        recorder,
        cv,
        triangulation,
        gantry,
        fine_align,
    ])
