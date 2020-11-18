# for more useful warnings
include(EnableCompilerWarnings)

#[[ [comment-willow-targets]

  Popart Targets Created Here
  ---------------------------

  We build three targets, which expose the dependencies to varying degrees of
  visibility.

  popart
    This is the main target most users should link against.
  
  popart-only
    Keeps onnx, protobuf, and the poplar libraries private. In general, these
    may be required by the user to actually interact usefully with the C++ API;
    but for some users (in particular poptorch) it is unnecessary and pollutes
    their package.
  
  popart-internal
    Has everything internally used as public. This is for internal targets
    only, like tests, so does not get installed (no config file).
  
  Look at the variables defined below for the exact definitions of what is
  public and private at each visibility level.
  
  Implementation Notes for Maintainers:
  -------------------------------------

    You can think of the visibility levels as:
      
        popart-only < popart < popart-internal
    
    Target B will have A in its link interface, if A < B as described above.

    The lowest visibility target will be the only actual library. It will
    include and link everything as required to be built, keeping things at the
    desired (lowest) level of visibility. The other targets will all be
    INTERFACE targets that incrementally add more dependencies to the interface
    properties.

    The variables POPART_PRIVATE_DEPS_* are private to popart (all targets). If
    a dependency is private even at the lowest visibility level, it should be
    added once here.

    The variables <target>_DEPS_* are public to that target and above. If a
    dependency is public to the visibility level of <target> and above, it
    should be added once here.

    These variables, when passed to `target_include_directories` and
    `target_link_libraries`, will be wrapped in `$<BUILD_INTERFACE:...>` genexes
    because we do not ship them in the installed package - we expect the user
    to provide the dependencies themselves when linking popart.
]]

################################################################################
##### Find dependencies
################################################################################

# Note, some of the dependencies are "found" in the parent cmake file, some are
# found here.

# Versions are not given where packages don't support them.

# Link Boost statically
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(boost_components
    # FindBoost bug: finding filesystem component does not transitively find and
    # link the system component.
    system

    filesystem
    graph
    random
)
find_package(Boost 1.70 REQUIRED COMPONENTS ${boost_components})

list(TRANSFORM boost_components PREPEND Boost:: OUTPUT_VARIABLE boost_targets)

# The header-only component. Many of the libs are header-only, and come under
# this target. You do not pass it as a COMPONENT; it will always be created.
list(APPEND boost_targets Boost::boost)

find_package(spdlog 1.8.0 EXACT REQUIRED)

find_package(poplar CONFIG REQUIRED)
find_package(poplibs CONFIG REQUIRED)
find_package(libpvti CONFIG REQUIRED)
find_package(gcl MODULE REQUIRED)

find_package(poprithms CONFIG REQUIRED)

# Onnx's config file fails to find protobuf, so we have to do it ourselves.
# We use the find module provided by CMake.
find_package(Protobuf 3.6 MODULE REQUIRED)
find_package(ONNX 1.6 CONFIG REQUIRED)

find_package(PoplarRunner)

################################################################################
##### Capnp
################################################################################

set(CAPNPC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/capnp")

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

set(POPART_INTERNAL_CAPNP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/capnp)

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
target_include_directories(popart_capnp
  PUBLIC
    ${CAPNPC_OUTPUT_DIR}
  PRIVATE
    ${CAPNP_INCLUDE_DIRECTORY}
  )

add_dependencies(popart_capnp create_capnp_output_dir)
target_compile_features(popart_capnp PUBLIC cxx_std_14)


################################################################################
##### Dependecy visibility levels (see [comment-willow-targets])
################################################################################

set(POPART_PRIVATE_DEPS_INCLUDES
)
set(POPART_PRIVATE_DEPS_LIBS
    ${boost_targets}
    spdlog::spdlog_header_only
)

set(POPART_ONLY_DEPS_INCLUDES
)
set(POPART_ONLY_DEPS_LIBS
)

set(POPART_DEPS_INCLUDES
    ${CAPNP_INCLUDE_DIRECTORY}
)
set(POPART_DEPS_LIBS
    poplar
    poplin
    popnn
    popops
    poprand
    poputil
    libpvti
    gcl
    CapnProto::capnp
    CapnProto::capnp-json
    ${popart_capnp}
    onnx
    onnx_proto
)
if (PoplarRunner_FOUND)
  list(APPEND POPART_DEPS_LIBS PoplarRunner::PoplarExecutableData)
endif()

set(POPART_INTERNAL_DEPS_INCLUDES
)
set(POPART_INTERNAL_DEPS_LIBS
  poprithms
)

################################################################################
##### Sources
################################################################################

# core popart source files:
file(GLOB core_sources src/*cpp)
file(GLOB op_sources src/op/*cpp)
file(GLOB op_dyn_sources src/op/dynamic/*cpp)
file(GLOB op_collectives_sources src/op/collectives/*cpp)
file(GLOB ces_sources src/ces/*cpp)
file(GLOB patterns_sources src/patterns/*cpp)
file(GLOB transforms_sources src/transforms/*cpp)
file(GLOB subgraph_sources src/subgraph/*cpp)

# poplar backend op specifc:
file(GLOB x_sources src/popx/*cpp)
file(GLOB opx_sources src/popx/op/*cpp)
file(GLOB opx_dyn_sources src/popx/op/dynamic/*cpp)
file(GLOB opx_collectives_sources src/popx/op/collectives/*cpp)

set(sources
  ${core_sources}
  ${op_sources}
  ${op_dyn_sources}
  ${op_collectives_sources}
  ${ces_sources}
  ${patterns_sources}
  ${transforms_sources}
  ${subgraph_sources})

# concatenate poplar backend sources to the list of files to compile
set(sources
  ${sources}
  ${x_sources}
  ${opx_sources}
  ${opx_dyn_sources}
  ${opx_collectives_sources})

# Fed to version header.
execute_process(COMMAND
  git rev-parse --short=10 HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(
    include/popart/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/popart/version.hpp
    @ONLY
)

################################################################################
##### Build targets
################################################################################

# IMPORTANT NOTE:
# A generator expression that wraps a list needs to be wrapped in quotes when
# passed to a command. Like:
#
#   target_include_directories(tgt "$<BUILD_INTERFACE:${some_list}>")
#
# See https://stackoverflow.com/questions/44425257/how-to-properly-use-target-include-directories-with-lists-of-includes

# The actual popart library we build. This is the lowest-visibility target.
# See [comment-willow-targets].

add_library(popart-only SHARED ${sources})

set_target_properties(popart-only PROPERTIES
    # We still want the actual library to be called, for example on linux,
    # libpopart.so, not libpopart-only.so.
    OUTPUT_NAME popart

    # Need position indepenent code to make dyanamic library with static libs.
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(popart-only PUBLIC cxx_std_14)

# These definitions are private to popart-only, as the external user does not
# need them, but they are needed by all internal targets.
set(popart_common_definitions
    # This allows us to deprecate the legacy op factory functions in our own
    # code without breaking customers code.
    DEPRECATE_LEGACY_OP_FACTORY=1
)

target_compile_definitions(popart-only
  PRIVATE
    # Required to prevent warnings.
    PROTOBUF_INLINE_NOT_IN_HEADERS=0

    ${popart_common_definitions}
)

if (PoplarRunner_FOUND)
  target_compile_definitions(popart-only PUBLIC POPLAR_RUNNER)
endif()

target_include_directories(popart-only
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/willow/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>

    # For generated headers like version.hpp.in
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE
    # Include headers in src privately (not part of public API).
    ${PROJECT_SOURCE_DIR}/willow/src
)

# Popart headers do not fully hide poplar headers. With our current lack of
# public-private separation, trying to fix places we leak poplar and preventing
# active development from re-leaking, will be very flakey.
#
# Thus, make the poplar headers public again in popart-only. Users of
# popart-only will still have to link poplar themselves, and hope they have a
# compatible version.
target_include_directories(popart-only PUBLIC $<BUILD_INTERFACE:${POPLAR_INCLUDE_DIR}>)

# Link dependencies, exposed at varying levels of visibility through interface
# targets. See [comment-willow-targets].

target_link_libraries(popart-only
  PUBLIC
    "$<BUILD_INTERFACE:${POPART_ONLY_DEPS_LIBS}>"
  PRIVATE
    ${POPART_PRIVATE_DEPS_LIBS}
    ${POPART_DEPS_LIBS}
    ${POPART_INTERNAL_DEPS_LIBS}
)
target_include_directories(popart-only SYSTEM
  PUBLIC
    "$<BUILD_INTERFACE:${POPART_ONLY_DEPS_INCLUDES}>"
)
target_include_directories(popart-only SYSTEM
  PRIVATE
    ${POPART_PRIVATE_DEPS_INCLUDES}
    ${POPART_DEPS_INCLUDES}
    ${POPART_INTERNAL_DEPS_INCLUDES}
)

add_dependencies(popart-only popart_capnp)

add_library(popart INTERFACE)
target_link_libraries(popart INTERFACE popart-only "$<BUILD_INTERFACE:${POPART_DEPS_LIBS}>")
target_include_directories(popart SYSTEM INTERFACE "$<BUILD_INTERFACE:${POPART_DEPS_INCLUDES}>")

add_library(popart-internal INTERFACE)
target_link_libraries(popart-internal INTERFACE popart "$<BUILD_INTERFACE:${POPART_INTERNAL_DEPS_LIBS}>")
target_include_directories(popart-internal SYSTEM INTERFACE "$<BUILD_INTERFACE:${POPART_INTERNAL_DEPS_INCLUDES}>")

target_include_directories(popart-internal INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/willow/src>")

target_compile_definitions(popart-internal INTERFACE ${popart_common_definitions})

################################################################################
##### Install popart and popart-only
################################################################################

# Note, the GNUInstallDirs are cache variables that the user can override.
# We make the config install dir overridable too.
set(POPART_CMAKE_INSTALL_CONFIGDIR "${CMAKE_INSTALL_LIBDIR}/cmake/popart"
    CACHE PATH 
    "Install directory for Popart's CMake config files. Defaults to \
    ${CMAKE_INSTALL_LIBDIR}/cmake/popart, where CMAKE_INSTALL_LIBDIR is \
    generated by the CMake package GNUInstallDirs."
)

install(TARGETS popart-only popart
    EXPORT popart-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY "include/popart"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT popart-targets
    FILE popart-targets.cmake
    DESTINATION ${POPART_CMAKE_INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    popart-config.cmake.in
    popart-config.cmake
    INSTALL_DESTINATION ${POPART_CMAKE_INSTALL_CONFIGDIR}
    PATH_VARS
      CMAKE_INSTALL_INCLUDEDIR
      CMAKE_INSTALL_LIBDIR
      POPART_CMAKE_INSTALL_CONFIGDIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/popart-config-version.cmake
  VERSION ${POPART_VERSION}
  COMPATIBILITY SameMinorVersion
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/popart-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/popart-config-version.cmake"
        DESTINATION ${POPART_CMAKE_INSTALL_CONFIGDIR}
)

# See T6089: to use python3 where strictly required, and
# then remove this commented out code:
#> add_custom_target(
#>         python_ver_check
#>         COMMAND python -c "import sys; sys.exit(sys.version_info.major <= 2)"
#>         VERBATIM
#> )
#> add_dependencies(popart python_ver_check)
