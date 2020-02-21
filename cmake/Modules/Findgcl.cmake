set(GCL_HINT_PATHS ${POPLAR_INSTALL_DIR}/gcl/include ${POPLAR_INSTALL_DIR}/include)
if(gcl_FOUND)
 return()
endif()

find_library(GCL_LIBRARIES
  NAMES gcl_ct
  HINTS ${POPLAR_INSTALL_DIR}/gcl/lib ${POPLAR_INSTALL_DIR}/lib
  DOC "gcl library to link to"
)
if(NOT GCL_LIBRARIES)
  message(FATAL_ERROR "Could not find gcl lib.")
endif()
message(STATUS "Found GCL_LIBRARIES ${GCL_LIB}")
mark_as_advanced(GCL_LIBRARIES)

find_path(GCL_INCLUDE_DIR gcl/ct/TileAllocation.hpp HINT ${GCL_HINT_PATHS})
set(GCL_INCLUDE_DIRS ${GCL_INCLUDE_DIR})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(gcl DEFAULT_MSG GCL_INCLUDE_DIR)
