set(WhatToDoString "Try setting PoplarRunner_INSTALL_DIR if not already done, \
something like -DPoplarRunner_INSTALL_DIR=/path/to/poplar/runner/, \
otherwise additional search paths might be needed in FindPoplarRunner.cmake \
(add to HINTS and/or PATH_SUFFIXES)")

FIND_LIBRARY(PoplarRunner_LIB
  NAMES poplar_executable_data
  HINTS ${PoplarRunner_INSTALL_DIR}
  PATH_SUFFIXES lib
  DOC "poplar_executable_data library to link to (for offline compilation / export)")
IF(NOT PoplarRunner_LIB)
  MESSAGE(WARNING "Could not set PoplarRunner_LIB, ${WhatToDoString}")
ENDIF()

MESSAGE(STATUS "found poplar_runner, defining PoplarRunner_LIB: ${PoplarRunner_LIB} (for offline compilation / export)")

FIND_PATH(PoplarRunner_INCLUDE_DIR
  NAMES ipu/poplar_executable_data.h
  PATHS ${PoplarRunner_INSTALL_DIR}
  PATH_SUFFIXES include
  DOC "directory containing the Poplar Runner headers.")
IF(NOT PoplarRunner_INCLUDE_DIR)
  MESSAGE(WARNING "Could not set PoplarRunner_INCLUDE_DIR, ${WhatToDoString}")
ENDIF()
MESSAGE(STATUS "found ipu/poplar_executable_data.h, defining \
PoplarRunner_INCLUDE_DIR: ${PoplarRunner_INCLUDE_DIR}")
MARK_AS_ADVANCED(PoplarRunner_INCLUDE_DIR PoplarRunner_LIB)

IF(PoplarRunner_INCLUDE_DIR)
  IF(NOT TARGET PoplarRunner::PoplarExecutableData)
    add_library(PoplarRunner::PoplarExecutableData UNKNOWN IMPORTED)
    set_target_properties(PoplarRunner::PoplarExecutableData PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${PoplarRunner_INCLUDE_DIR}"
      IMPORTED_LOCATION "${PoplarRunner_LIB}"
      )
  endif()
endif()

include(FindPackageHandleStandardArgs)
# Sets sxpconfig_FOUND
find_package_handle_standard_args(PoplarRunner DEFAULT_MSG PoplarRunner_INCLUDE_DIR PoplarRunner_LIB)
