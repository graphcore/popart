# This compiles a shared object file for the cube operator.
# Run make in the custom_ops folder to build. 
cmake_minimum_required(VERSION 3.4)

project(cube.so)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS ON)

# All C++ code in this project will be compiled as C++14
set (CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#Where to find Findxxxx.cmake files
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/Modules ${PROJECT_SOURCE_DIR}/../../cmake)


#with cmake >= 3.12 this should be add_compile_definitions(ONNX_NAMESPACE=onnx)
add_definitions(-DONNX_NAMESPACE=onnx)

# Using installed cmake find (ONNX must be installed)
# For a hand-crafted Findonnx script, see a very early commit
# or repo popart if available
message(STATUS "Linking Onnx...")
find_package(ONNX REQUIRED)

# Add the shared library to the project
add_library(cube SHARED custom_op.cpp)

# Copy the python script to build folder
configure_file(custom_op.py custom_op.py COPYONLY)

# Link all the required libraries. If you do not require all the Poplar 
# (poplin, popops etc) libraries, you can remove them without issue.
# However, PopART and Poplar will be required in all cases.
message(STATUS "Linking PopART...")
target_link_libraries(cube popart)
message(STATUS "Linking Poplar...")
target_link_libraries(cube poplar)
message(STATUS "Linking Popops...")
target_link_libraries(cube popops)
message(STATUS "Linking Poplin...")
target_link_libraries(cube poplin)
message(STATUS "Linking Popnn...")
target_link_libraries(cube popnn)
message(STATUS "Linking Poprand...")
target_link_libraries(cube poprand)
