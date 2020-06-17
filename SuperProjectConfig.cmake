include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL
    "Default value for POSITION_INDEPENDENT_CODE of targets.")

set(POPLAR_INSTALL_DIR "" CACHE STRING "The Poplar install directory")
list(APPEND POPART_CMAKE_ARGS -DPOPLAR_INSTALL_DIR=${POPLAR_INSTALL_DIR})

set(PoplarRunner_INSTALL_DIR "" CACHE STRING "The Poplar Runner install directory")
list(APPEND POPART_CMAKE_ARGS -DPoplarRunner_INSTALL_DIR=${PoplarRunner_INSTALL_DIR})

set(C10_DIR "" CACHE STRING "Directory to install Cifar-10 dataset to")
list(APPEND POPART_CMAKE_ARGS -DC10_DIR=${C10_DIR})

set(CMAKE_CONFIGURATION_TYPES "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
# Enable IN_LIST operator
cmake_policy(SET CMP0057 NEW)
if(NOT CMAKE_BUILD_TYPE)
  list(GET CMAKE_CONFIGURATION_TYPES 0 CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
endif()
if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of ${CMAKE_CONFIGURATION_TYPES}")
endif()
