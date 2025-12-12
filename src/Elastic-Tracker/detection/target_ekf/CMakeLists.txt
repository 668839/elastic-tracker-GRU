
cmake_minimum_required(VERSION 3.5)
project(target_ekf)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  object_detection_msgs
  nav_msgs
  message_filters
  Eigen3 REQUIRED
  # 新增
  gazebo_msgs
)


catkin_package(
  INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR}
  CATKIN_DEPENDS object_detection_msgs
  DEPENDS EIGEN3
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

add_executable(target_ekf_node
 src/target_ekf_node.cpp
)

target_link_libraries(target_ekf_node
   ${catkin_LIBRARIES}
)

add_dependencies(target_ekf_node
  ${catkin_EXPORTED_TARGETS}
)


# 新增：Gazebo目标位置发布节点
add_executable(gazebo_target_node
  src/gazebo_target_node.cpp
)

target_link_libraries(gazebo_target_node
  ${catkin_LIBRARIES}
)

add_dependencies(gazebo_target_node
  ${catkin_EXPORTED_TARGETS}
)