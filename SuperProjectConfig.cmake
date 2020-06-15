include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL
    "Default value for POSITION_INDEPENDENT_CODE of targets.")

set(POPLAR_INSTALL_DIR "" CACHE STRING "The Poplar install directory")
list(APPEND POPART_CMAKE_ARGS -DPOPLAR_INSTALL_DIR=${POPLAR_INSTALL_DIR})

set(PoplarRunner_INSTALL_DIR "" CACHE STRING "The Poplar Runner install directory")
list(APPEND POPART_CMAKE_ARGS -DPoplarRunner_INSTALL_DIR=${PoplarRunner_INSTALL_DIR})

set(C10_DIR "" CACHE STRING "Directory to install Cifar-10 dataset to")
list(APPEND POPART_CMAKE_ARGS -DC10_DIR=${C10_DIR})
