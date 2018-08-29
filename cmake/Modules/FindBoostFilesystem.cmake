FIND_LIBRARY(BOOST_SYSTEM_LIB
  NAMES boost_system
  HINTS ${POPLAR_INSTALL_DIR}/boost
  PATH_SUFFIXES boost boost/lib
  # note : paths checked after hints, so boost on pop install path has preference
  PATHS /usr/local/lib 
  DOC "boost_system lib")
IF(NOT BOOST_SYSTEM_LIB)
  MESSAGE(FATAL_ERROR "Could not set BOOST_SYSTEM_LIB, set POPLAR_INSTALL_DIR (-DPOPLAR_INSTALL_DIR=...) and if boost is installed there it should be found")
ELSE()
  MESSAGE(STATUS "BOOST_SYSTEM_LIB set to ${BOOST_SYSTEM_LIB}")
ENDIF()
MARK_AS_ADVANCED(BOOST_SYSTEM_LIB)


FIND_LIBRARY(BOOST_FILESYSTEM_LIB
  NAMES boost_filesystem
  HINTS ${POPLAR_INSTALL_DIR}/boost
  PATH_SUFFIXES boost boost/lib
  PATHS /usr/local/lib
  DOC "boost_filesystem lib")
IF(NOT BOOST_FILESYSTEM_LIB)
  MESSAGE(FATAL_ERROR "Could not set BOOST_FILESYSTEM_LIB, set POPLAR_INSTALL_DIR (-DPOPLAR_INSTALL_DIR=...) and if boost is installed there it should be found")
ELSE()
  MESSAGE(STATUS "BOOST_FILESYSTEM_LIB set to ${BOOST_FILESYSTEM_LIB}")
ENDIF()
MARK_AS_ADVANCED(BOOST_FILESYSTEM_LIB)


FIND_PATH(BOOST_FILESYSTEM_INCLUDE_DIR
  NAMES boost/filesystem.hpp
  HINTS ${POPLAR_INSTALL_DIR}/boost/include
  PATH_SUFFIXES boost boost/include boost/include/boost include
  #note : paths tried after hints
  PATHS /usr/local/Cellar/boost/1.66.0/include
  DOC "directory with poplar boost include files (boost/filesystem.hpp etc.)")
IF(NOT BOOST_FILESYSTEM_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Could not set BOOST_FILESYSTEM_INCLUDE_DIR, set POPLAR_INSTALL_DIR")
ENDIF()
MARK_AS_ADVANCED(BOOST_FILESYSTEM_INCLUDE_DIR)



