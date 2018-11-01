cmake_minimum_required(VERSION 3.4)

project(willow)

set(INSTALL_PYDIR ${CMAKE_INSTALL_PREFIX}/python)
set(INSTALL_EXAMPLES ${CMAKE_INSTALL_PREFIX}/examples)

#with cmake >= 3.12 this should be add_compile_definitions(ONNX_NAMESPACE=onnx)
add_definitions(-DONNX_NAMESPACE=onnx)

#Where to find Findxxxx.cmake files
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/cmake)

find_package(Threads REQUIRED)

# There is a FindProtobuf.cmake file on my computer at
# /usr/local/share/cmake/Modules/FindProtobuf.cmake
find_package(Protobuf REQUIRED) 

# Using installed cmake find (ONNX must be installed)
# For a hand-crafted Findonnx script, see a very early commit
# or repo poponnx if available
find_package(ONNX REQUIRED)

# Boost 1.66 and cmake do not agree. Waiting for fix from KitWARE.
# find_package(Boost COMPONENTS filesystem system REQUIRED) 
# For now I'm using a workaround search which searches along
# the poplar install path, and then on system path
find_package(BoostFilesystem REQUIRED) 

# at some future point we might want this for a CPU backend for willow
# find_package(CBLAS REQUIRED)

set(PYWILLOW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pywillow)
add_subdirectory(willow)

# light-weight wrapper using pybind11
add_subdirectory(pywillow)

# Examples and tests
add_subdirectory(tests)

