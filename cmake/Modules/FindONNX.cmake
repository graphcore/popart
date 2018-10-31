# currently link to protobuf generated files only. 
# there will also be the option to link to the onnx utils for
# model verification, to be added at a later point.

FIND_LIBRARY(ONNX_LIBRARIES
  NAMES onnx
  HINTS # ... to be extended if there's a commonly used path we should hit
  DOC "onnx library")
IF(NOT ONNX_LIBRARIES)
  MESSAGE(FATAL_ERROR "\
  Could not set ONNX_LIBRARIES. Consider adding HINTS or use CMAKE_PREFIX_PATH")
ENDIF()

MARK_AS_ADVANCED(ONNX_LIBRARIES)
MESSAGE(STATUS "found onnx ${ONNX_LIBRARIES}")

FIND_PATH(ONNX_INCLUDE_DIRS
  NAMES onnx/onnx_pb.h
  HINTS #
  PATH_SUFFIXES # onnx onnx/include
  DOC "dir with the generated header, onnx.pb.h (we use onnx_pb.h).")
IF(NOT ONNX_INCLUDE_DIRS)
  MESSAGE(FATAL_ERROR "Could not set ONNX_INCLUDE_DIRS. \
  Consider adding HINTS and PATH_SUFFIXES,  or use CMAKE_PREFIX_PATH")
ENDIF()

MARK_AS_ADVANCED(ONNX_INCLUDE_DIRS)
MESSAGE(STATUS "found onnx/onnx_pb.h, defining \
ONNX_INCLUDE_DIRS: ${ONNX_INCLUDE_DIRS}")

SET(ONNX_FIND_QUIETLY OFF)
IF (NOT ONNX_FIND_QUIETLY)
  MESSAGE(STATUS "Found onnx library and header")
ENDIF()
