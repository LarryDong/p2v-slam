cmake_minimum_required(VERSION 3.10)
project(p2v-slam)
# -------------------------
# Build type & C++ standard
# -------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)   # release mode, faster.
  add_compile_options(-O0 -DNDEBUG)
  # add_compile_options(-O3 -DNDEBUG)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-fexceptions -pthread)
add_compile_options(-DBOOST_BIND_GLOBAL_PLACEHOLDERS) # ignore BOOST warning

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message(STATUS "Processor number: ${N}")
  if(N GREATER 14)
    add_definitions(-DMP_EN -DMP_PROC_NUM=12)
    message(STATUS "core for MP: 12")
  elseif(N GREATER 10)
    add_definitions(-DMP_EN -DMP_PROC_NUM=8)
    message(STATUS "core for MP: 8")
  elseif(N GREATER 7)
    add_definitions(-DMP_EN -DMP_PROC_NUM=6)
    message(STATUS "core for MP: 6")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN -DMP_PROC_NUM=3)
    message(STATUS "core for MP: 3")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()


add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
# -------------------------
# Dependencies
# -------------------------
find_package(Boost REQUIRED COMPONENTS timer)
find_package(OpenMP QUIET)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common kdtree filters)   # TODO: PCL version may be modified.



# libtorch
# TODO: You need to install libtorch first, and give the libtorch's cmake
set(Torch_DIR "/home/larry/codeGit/libtorch/share/cmake/Torch")   # set to your own path/to/libtorch
find_package(Torch REQUIRED)


# onnxruntime
# TODO: You need to install onnx-runtime first and give the path.
set(ONNXRUNTIME_ROOT "/home/larry/codeGit/onnxruntime" CACHE PATH "onnxruntime root")   # set to your own path/to/onnxruntime
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
  message(FATAL_ERROR "ONNXRUNTIME_ROOT is invalid: ${ONNXRUNTIME_ROOT}. Expect ${ONNXRUNTIME_ROOT}/include")
endif()
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)


# ROS (catkin)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)


generate_messages(
  DEPENDENCIES
  geometry_msgs
)
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
)


include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)


add_executable(p2v_node
  src/voxelMapping.cpp
  src/preprocess.cpp
  src/voxel_map_util.cpp
  src/IMU_Processing.cpp
  src/implicit_voxel_map.cpp
  src/p2v_model.cpp
  src/viewer.cpp
  src/PointCloudUn.cpp
  src/ObsScheduler.cpp
)
target_include_directories(p2v_node PRIVATE
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
# OpenMP 目标链接
if(OpenMP_CXX_FOUND)
  target_link_libraries(p2v_node PRIVATE OpenMP::OpenMP_CXX)
endif()


target_link_libraries(p2v_node
  PRIVATE
    ${catkin_LIBRARIES}
    ${TORCH_LIBRARIES}
    onnxruntime
    pcl_common
    pcl_kdtree
    pcl_filters
)
