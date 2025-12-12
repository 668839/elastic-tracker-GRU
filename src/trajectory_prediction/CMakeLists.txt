cmake_minimum_required(VERSION 3.5)
project(trajectory_prediction)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  message_generation
)

# 自定义消息
add_message_files(
  FILES
  TrajectoryPrediction.msg
  PredictionWithUncertainty.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs nav_msgs geometry_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# 数据收集节点 (C++)
add_executable(data_collector_node src/data_collector_node.cpp)
target_link_libraries(data_collector_node ${catkin_LIBRARIES})
add_dependencies(data_collector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 安装Python脚本
catkin_install_python(PROGRAMS
  # 数据收集
  scripts/collect_gazebo_actor.py
  # 数据预处理
  scripts/preprocess_data.py
  scripts/preprocess_gazebo_data.py
  # 模型训练与评估
  scripts/train_teacher.py
  scripts/evaluate_model.py
  # 工具脚本
  scripts/generate_synthetic_data.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
