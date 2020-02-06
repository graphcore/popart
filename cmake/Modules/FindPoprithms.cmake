set(WhatToDoString "Set POPRITHMS_INSTALL_DIR, \
something like -DPOPRITHMS_INSTALL_DIR=/path/to/build/install/")

FIND_PATH(POPRITHMS_INCLUDE_DIR 
  NAMES poprithms/schedule/anneal/graph.hpp
  HINTS ${POPRITHMS_INSTALL_DIR} ${POPRITHMS_INSTALL_DIR}/include 
  PATH_SUFFIXES poprithms poprithms/include
  DOC "directory with poprithms include files (poprithms/schedule/anneal/graph.hpp etc.)")
IF(NOT POPRITHMS_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Could not set POPRITHMS_INCLUDE_DIR. ${WhatToDoString}")
ENDIF()
MESSAGE(STATUS "Found POPRITHMS_INCLUDE_DIR ${POPRITHMS_INCLUDE_DIR}")
MARK_AS_ADVANCED(POPRITHMS_INCLUDE_DIR)

FIND_LIBRARY(POPRITHMS_LIB
  NAMES poprithms
  HINTS ${POPRITHMS_INSTALL_DIR}/poprithms/lib ${POPRITHMS_INSTALL_DIR}/lib
  PATH_SUFFIXES poprithms poprithms/lib
  DOC "poprithms library to link to")
IF(NOT POPRITHMS_LIB)
  MESSAGE(FATAL_ERROR "Could not set POPRITHMS_LIB. ${WhatToDoString}")
ENDIF()
MESSAGE(STATUS "Found POPRITHMS_LIB ${POPRITHMS_LIB}")
MARK_AS_ADVANCED(POPRITHMS_LIB)
