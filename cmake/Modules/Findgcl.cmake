# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
if(gcl_FOUND)
  return()
endif()

find_library(gcl_ct_LIB
  NAMES gcl gcl_ct
  HINTS ${POPLAR_INSTALL_DIR}/gcl/lib ${POPLAR_INSTALL_DIR}/lib
  DOC "gcl library to link to"
)

find_path(gcl_INCLUDE_DIR
  NAMES gcl/TileAllocation.hpp
  HINTS ${POPLAR_INSTALL_DIR}/gcl/include ${POPLAR_INSTALL_DIR}/include
  DOC "gcl include dir"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(gcl DEFAULT_MSG gcl_ct_LIB gcl_INCLUDE_DIR)

# Everything has been found correctly, create the variables and targets that the
# user will use.

set(GCL_LIBRARIES "${gcl_ct_LIB}")
set(GCL_INCLUDE_DIRS "${gcl_INCLUDE_DIR}")

add_library(gcl SHARED IMPORTED)

set_target_properties(gcl PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${GCL_INCLUDE_DIRS}"
  IMPORTED_LOCATION "${gcl_ct_LIB}"
)

message(STATUS "Created imported library gcl using gcl_ct_LIB and GCL_INCLUDE_DIRS")
