set(WhatToDoString "Try setting PROTOBUF_INSTALL_DIR if not already done, \
something like -DPROTOBUF_INSTALL_DIR=/path/to/protobuf/build/install/, \
otherwise additional search paths might be needed in FindPROTOBUF.cmake \
(add to HINTS and/or PATH_SUFFIXES)")

MESSAGE(STATUS "PROTOBUF:${PROTOBUF_INSTALL_DIR} ONNX:${ONNX_INSTALL_DIR}")

FIND_PATH(PROTOBUF_INCLUDE_DIR
  NAMES google/protobuf/port_def.inc
  HINTS ${PROTOBUF_INSTALL_DIR}/include 
  PATH_SUFFIXES
  DOC "directory containing the proto buf header, google/protobuf/port_def.inc")
IF(NOT PROTOBUF_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Could not set PROTOBUF_INCLUDE_DIR, ${WhatToDoString}")
ENDIF()
MARK_AS_ADVANCED(PROTOBUF_INCLUDE_DIR)
MESSAGE(STATUS "found google/protobuf/port_def.inc, defining PROTOBUF_INCLUDE_DIR: ${PROTOBUF_INCLUDE_DIR}")

FIND_LIBRARY(PROTOBUF_LIB
  NAMES protobuf
  HINTS ${PROTOBUF_INSTALL_DIR}/lib
  NO_CMAKE_SYSTEM_PATH
  DOC "protobuf library to link to")
  
IF(NOT PROTOBUF_LIB)
  MESSAGE(FATAL_ERROR "Could not set PROTOBUF_LIB, ${WhatToDoString}")
ENDIF()
MARK_AS_ADVANCED(PROTOBUF_LIB)
MESSAGE(STATUS "found protobuf, defining PROTOBUF_LIB: ${PROTOBUF_LIB}")


