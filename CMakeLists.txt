cmake_minimum_required(VERSION 3.0)

project(neuralnet)

#Where to find FindALib.cmake files
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/cmake)

find_package(Threads REQUIRED)

# I have written a FindONNX.cmake file
find_package(onnx REQUIRED)

# There is a FindProtobuf.cmake file on my computer at
# /usr/local/share/cmake/Modules/FindProtobuf.cmake
find_package(Protobuf REQUIRED) 

# Boost 1.66 and cmake do not agree. Waiting for fix from KitWARE.
# find_package(Boost COMPONENTS filesystem system REQUIRED) 
# For now I'm using a workaround search which searches along
# the poplar install path, and then on system path
find_package(BoostFilesystem REQUIRED) 

#Does require that the used has set path, something like
find_package(CBLAS REQUIRED)

add_subdirectory(neuralnet)
