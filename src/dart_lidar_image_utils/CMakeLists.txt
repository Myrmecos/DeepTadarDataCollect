cmake_minimum_required(VERSION 3.0.2)
project(dart_lidar_image_utils)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES dart_lidar_image_utils
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
