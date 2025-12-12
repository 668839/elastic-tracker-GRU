cmake_minimum_required(VERSION 3.5)
project(gazebo_target_tracker)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  geometry_msgs
  gazebo_msgs
  mavros_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
  CATKIN_DEPENDS roscpp std_msgs nav_msgs geometry_msgs gazebo_msgs mavros_msgs tf2 tf2_ros tf2_geometry_msgs
  DEPENDS EIGEN3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# 主跟踪节点
add_executable(gazebo_target_tracker_node src/gazebo_target_tracker_node.cpp)
target_link_libraries(gazebo_target_tracker_node ${catkin_LIBRARIES})
add_dependencies(gazebo_target_tracker_node ${catkin_EXPORTED_TARGETS})

# 简单控制器节点
add_executable(simple_follower_node src/simple_follower_node.cpp)
target_link_libraries(simple_follower_node ${catkin_LIBRARIES})
add_dependencies(simple_follower_node ${catkin_EXPORTED_TARGETS})
