# currently link to protobuf generated files only. 
# there will also be the option to link to the onnx utils for
# model verification, to be added at a later point.

FIND_LIBRARY(ONNX_PROTO_LIB
  NAMES onnx_proto
  HINTS ${ONNX_ROOT_DIR}/.setuptools-cmake-build
  DOC "onnx library to link to (corresponding to prot generated header)")
IF(NOT ONNX_PROTO_LIB)
  MESSAGE(FATAL_ERROR "Could not set ONNX_PROTO_LIB, \
  set ONNX_ROOT_DIR (-DONNX_ROOT_DIR=...)")
ENDIF()
MARK_AS_ADVANCED(ONNX_PROTO_LIB)
MESSAGE(STATUS "found onnx_proto ${ONNX_PROTO_LIB}")


FIND_LIBRARY(ONNX_LIB
  NAMES onnx
  HINTS ${ONNX_ROOT_DIR}/.setuptools-cmake-build
  DOC "onnx library to link to (for model tests)")
IF(NOT ONNX_PROTO_LIB)
  MESSAGE(FATAL_ERROR "\
Could not set ONNX_LIB, set ONNX_ROOT_DIR (-DONNX_ROOT_DIR=...) \
if not already done, or else additional search paths might be \
needed in Findonnx.cmake (add to HINTS and/or PATH_SUFFIXES)")
ENDIF()
MARK_AS_ADVANCED(ONNX_LIB)
MESSAGE(STATUS "found onnx ${ONNX_LIB} (for model testing)")

FIND_PATH(ONNX_CHECKER_INCLUDE_DIR
  NAMES onnx/checker.h
  HINTS ${ONNX_ROOT_DIR}
  PATH_SUFFIXES  # onnx onnx/include onnx/onnx
  DOC "directory containing the onnx model checker header, checker.h")
IF(NOT ONNX_CHECKER_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "\
Could not set ONNX_CHECKER_INCLUDE_DIR, \
set ONNX_ROOT_DIR (-DONNX_ROOT_DIR=...) if not already done, \
otherwise Findonnx.cmake might need additional paths to search on")
ENDIF()
MARK_AS_ADVANCED(ONNX_CHECKER_INCLUDE_DIR)
MESSAGE(STATUS "found onnx/checker.h, defining \
ONNX_CHECKER_INCLUDE_DIR: ${ONNX_CHECKER_INCLUDE_DIR}")


FIND_PATH(ONNX_PB_INCLUDE_DIR
  NAMES onnx/onnx.pb.h
  HINTS ${ONNX_ROOT_DIR}/.setuptools-cmake-build
  PATH_SUFFIXES # onnx onnx/include
  DOC "directory containing the protobuf generated header, onnx.pb.h")
IF(NOT ONNX_PB_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "\
Could not set ONNX_PB_INCLUDE_DIR, \
set ONNX_ROOT_DIR (-DONNX_ROOT_DIR=...)")
ENDIF()
MARK_AS_ADVANCED(ONNX_PB_INCLUDE_DIR)
MESSAGE(STATUS "found onnx/onnx.pb.h, defining \
ONNX_PB_INCLUDE_DIR: ${ONNX_PB_INCLUDE_DIR}")


FIND_PATH(ONNX_SCHEMA_INCLUDE_DIR
  NAMES onnx/defs/schema.h
  HINTS ${ONNX_ROOT_DIR}
  PATH_SUFFIXES # onnx onnx/include
  DOC "directory containing the header with functions \
  for checking opset:version mapping, schema.h")
  IF(NOT ONNX_SCHEMA_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "Could not set ONNX_SCHEMA_INCLUDE_DIR, \
  set ONNX_ROOT_DIR (-DONNX_ROOT_DIR=...)")
ENDIF()
MARK_AS_ADVANCED(ONNX_SCHEMA_INCLUDE_DIR)
MESSAGE(STATUS "\
found onnx/defs/schema.h, defining \
ONNX_SCHEMA_INCLUDE_DIR: ${ONNX_SCHEMA_INCLUDE_DIR}")






SET(ONNX_FIND_QUIETLY OFF)
  IF (NOT ONNX_FIND_QUIETLY)
    MESSAGE(STATUS "Found onnx library and header")
  ENDIF()
