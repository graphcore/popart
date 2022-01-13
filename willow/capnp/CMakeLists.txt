# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
find_package(CapnProto REQUIRED)

add_custom_target(create_capnp_output_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CAPNPC_OUTPUT_DIR})

# Copied from Poplar CMakeLists.txt
function(popart_capnp_generate OUTPUT_DIR IMPORT_DIRS SRCS HDRS)
  if (TARGET capnp_tool)
    set(CAPNP_EXECUTABLE capnp_tool)
    get_target_property(CAPNPC_CXX_EXECUTABLE capnpc_cpp CAPNPC_EXECUTABLE)
    get_target_property(CAPNP_INCLUDE_DIRECTORY capnp_tool CAPNP_INCLUDE_DIRECTORY)
    list(APPEND tool_depends capnp_tool capnpc_cpp)
  endif()
  if(NOT CAPNP_EXECUTABLE)
    message(SEND_ERROR "Could not locate capnp executable (CAPNP_EXECUTABLE)")
  endif()
  if(NOT CAPNPC_CXX_EXECUTABLE)
    message(SEND_ERROR "Could not locate capnpc-c++ executable (CAPNPC_CXX_EXECUTABLE)")
  endif()
  if(NOT CAPNP_INCLUDE_DIRECTORY)
    message(SEND_ERROR "Could not locate capnp header files (CAPNP_INCLUDE_DIRECTORY)")
  endif()

  set(include_paths -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${CAPNP_INCLUDE_DIRECTORY})

  if(DEFINED ${IMPORT_DIRS})
    foreach(directory ${${IMPORT_DIRS}})
      get_filename_component(absolute_path "${directory}" ABSOLUTE)
      list(APPEND include_paths -I "${absolute_path}")
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  get_filename_component(output_dir "${${OUTPUT_DIR}}" ABSOLUTE)
  foreach(schema_file ${ARGN})
    get_filename_component(schema_file "${schema_file}" ABSOLUTE)
    if(NOT EXISTS "${schema_file}")
      message(FATAL_ERROR "Schema file '${schema_file}' does not exist!")
    endif()

    # Find the prefix to chop off
    get_filename_component(file_prefix "${schema_file}" DIRECTORY)
    get_filename_component(file_name "${schema_file}" NAME)
    set(output_base "${output_dir}/${file_name}")
    add_custom_command(
      OUTPUT "${output_base}.c++" "${output_base}.h"
      COMMAND "${CAPNP_EXECUTABLE}"
      ARGS compile
        -o ${CAPNPC_CXX_EXECUTABLE}:${output_dir}
        --src-prefix=${file_prefix}
        ${include_paths}
        ${CAPNPC_FLAGS}
        ${schema_file}
      DEPENDS "${schema_file}" ${tool_depends}
      COMMENT "Generating capnproto sources for schema ${schema_file}"
      VERBATIM
    )

    list(APPEND ${SRCS} "${output_base}.c++")
    list(APPEND ${HDRS} "${output_base}.h")
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

set(POPART_INTERNAL_CAPNP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Generate the header and source files and assign their paths to capnp_srcs
# and capnp_hdrs. Note that this generation is done at compile time - not
# configure time, so the output files needed to be explicitly depended on by
# any and all targets that use them or they may not be generated.
popart_capnp_generate(CAPNPC_OUTPUT_DIR POPART_INTERNAL_CAPNP_DIR capnp_srcs capnp_hdrs
  ${POPART_INTERNAL_CAPNP_DIR}/Executablex.capnp
  ${POPART_INTERNAL_CAPNP_DIR}/IrLowering.capnp
  ${POPART_INTERNAL_CAPNP_DIR}/Ir.capnp
)

add_library(popart_capnp OBJECT ${capnp_srcs} ${capnp_hdrs})

target_link_libraries(popart_capnp
  PUBLIC
    CapnProto::capnp
    CapnProto::capnp-json
)

add_dependencies(popart_capnp create_capnp_output_dir)
target_compile_features(popart_capnp PUBLIC cxx_std_14)
