cmake_minimum_required(VERSION 3.5)
project(simple_chase)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  mavros_msgs
  gazebo_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(gazebo_fake_localization src/gazebo_fake_localization.cpp)
target_link_libraries(gazebo_fake_localization ${catkin_LIBRARIES})

add_executable(simple_car_tracker src/simple_car_tracker.cpp)
target_link_libraries(simple_car_tracker ${catkin_LIBRARIES})
