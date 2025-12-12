cmake_minimum_required(VERSION 3.5)
project(gru_trajectory_prediction)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-Wall -Wextra)

# ============================================
# Find Packages - 不包含planning！
# ============================================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf
  eigen_conversions
  mapping              # ✅ 需要（碰撞检测）
  quadrotor_msgs       # ✅ 需要（地图消息）
  # planning           # ❌ 删除！不需要
)

find_package(Eigen3 REQUIRED)

# ============================================
# Catkin Package Configuration
# ============================================
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES 
    collision_checker
    trajectory_corrector
    gru_prediction_client
  CATKIN_DEPENDS 
    roscpp 
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    mapping
    quadrotor_msgs
    # planning         # ❌ 删除！
  DEPENDS 
    EIGEN3
)

# ============================================
# Include Directories
# ============================================
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ============================================
# Build C++ Libraries
# ============================================

# 1. Collision Checker Library
add_library(collision_checker
  src/collision_checker.cpp
)
target_link_libraries(collision_checker
  ${catkin_LIBRARIES}
)
add_dependencies(collision_checker
  ${catkin_EXPORTED_TARGETS}
)

# 2. Trajectory Corrector Library
add_library(trajectory_corrector
  src/trajectory_corrector.cpp
)
target_link_libraries(trajectory_corrector
  collision_checker
  ${catkin_LIBRARIES}
)
add_dependencies(trajectory_corrector
  ${catkin_EXPORTED_TARGETS}
)

# 3. GRU Prediction Client Library
add_library(gru_prediction_client
  src/gru_prediction_client.cpp
)
target_link_libraries(gru_prediction_client
  trajectory_corrector
  ${catkin_LIBRARIES}
)
add_dependencies(gru_prediction_client
  ${catkin_EXPORTED_TARGETS}
)

# ============================================
# Build Executable (for testing)
# ============================================
add_executable(gru_prediction_client_node
  src/gru_prediction_client.cpp
)
target_link_libraries(gru_prediction_client_node
  gru_prediction_client
  ${catkin_LIBRARIES}
)

# ============================================
# Install Python Scripts
# ============================================
catkin_install_python(PROGRAMS
  scripts/train/step1_train_teacher.py
  scripts/train/step2_distill_student.py
  scripts/train/step3_prune_model.py
  scripts/train/step4_quantize_model.py
  scripts/train/step5_convert_torchscript.py
  scripts/data/collect_gazebo_data.py
  scripts/inference/gru_server.py
  scripts/inference/kalman_fusion.py
  scripts/visualization/trajectory_visualizer.py
  scripts/visualization/uncertainty_visualizer.py
  scripts/visualization/performance_monitor.py
  scripts/visualization/marker_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ============================================
# Install Headers
# ============================================
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

# Install prediction header (for planning to use)
install(DIRECTORY include/prediction/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/../prediction
  FILES_MATCHING PATTERN "*.hpp"
)

# ============================================
# Install Libraries
# ============================================
install(TARGETS 
    collision_checker 
    trajectory_corrector
    gru_prediction_client
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

# ============================================
# Install Launch Files
# ============================================
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# ============================================
# Install RViz Config
# ============================================
install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

# ============================================
# Install Config
# ============================================
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)