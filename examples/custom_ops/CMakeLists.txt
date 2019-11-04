# This compiles a shared object file for the cube operator.
# Run make in the custom_ops folder to build. 
cmake_minimum_required(VERSION 3.4)

project(custom_op.so)

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

# Add the executable to the project.
add_executable(custom_op.so custom_op.cpp)

# Link all the required libraries. If you do not require all the Poplar 
# (poplin, popops etc) libraries, you can remove them without issue.
# However, PopART and Poplar will be required in all cases.
message(STATUS "Linking PopART...")
target_link_libraries(custom_op.so popart)
message(STATUS "Linking Poplar...")
target_link_libraries(custom_op.so poplar)
message(STATUS "Linking Popops...")
target_link_libraries(custom_op.so popops)
message(STATUS "Linking Poplin...")
target_link_libraries(custom_op.so poplin)
message(STATUS "Linking Popnn...")
target_link_libraries(custom_op.so popnn)
message(STATUS "Linking Poprand...")
target_link_libraries(custom_op.so poprand)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
