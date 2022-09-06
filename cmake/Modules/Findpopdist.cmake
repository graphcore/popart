# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
if(popdist_FOUND)
  return()
endif()

find_library(popdist_LIB
  NAMES popdist
  HINTS ${POPLAR_INSTALL_DIR}/popdist/lib ${POPLAR_INSTALL_DIR}/lib
  DOC "popdist library to link to"
)

find_path(popdist_INCLUDE_DIR
  NAMES popdist/backend.hpp popdist/collectives.hpp popdist/context.hpp
  HINTS ${POPLAR_INSTALL_DIR}/popdist/include ${POPLAR_INSTALL_DIR}/include
  DOC "popdist include dir"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(popdist DEFAULT_MSG popdist_LIB popdist_INCLUDE_DIR)

# Everything has been found correctly, create the variables and targets that the
# user will use.

set(POPDIST_LIBRARIES "${popdist_LIB}")
set(POPDIST_INCLUDE_DIRS "${popdist_INCLUDE_DIR}")

add_library(popdist SHARED IMPORTED)

set_target_properties(popdist PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${POPDIST_INCLUDE_DIRS}"
  IMPORTED_LOCATION "${popdist_LIB}"
)

message(STATUS "Created imported library popdist using popdist_LIB and POPDIST_INCLUDE_DIRS")
