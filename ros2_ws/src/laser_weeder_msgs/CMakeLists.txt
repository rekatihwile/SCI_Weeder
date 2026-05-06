cmake_minimum_required(VERSION 3.8)
project(laser_weeder_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionList.msg"
  "msg/StereoTarget.msg"
  "msg/TargetList.msg"
  "msg/GantryStatus.msg"
  "msg/GantryCommand.msg"
  "msg/PixelError.msg"
  "msg/FineAlignGoal.msg"
  DEPENDENCIES std_msgs
)

ament_package()
