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
    The Poplar SDK libs are still public, but keeps onnx, protobuf private. In
    general, these may be required by the user to actually interact usefully
    with the C++ API; but for some users (in particular poptorch) it is
    unnecessary.
  
  popart-internal
    Has everything internally used as public. This is for internal targets
    only, like tests, so does not get installed (not in the config file).
  
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

# Creates popart_capnp object library from generated sources.
set(CAPNPC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/popart/capnp")
add_subdirectory(capnp)

################################################################################
##### Dependency visibility levels (see [comment-willow-targets])
################################################################################

set(POPART_PRIVATE_DEPS_INCLUDES
)
set(POPART_PRIVATE_DEPS_LIBS
    ${boost_targets}
    spdlog::spdlog_header_only
    popart_capnp
)

set(POPART_ONLY_DEPS_INCLUDES
)
set(POPART_ONLY_DEPS_LIBS
    poplar
    poplin
    popnn
    popops
    poprand
    poputil
    poprithms
    libpvti
    pva
    gcl
    snap
    gccs
)

set(POPART_DEPS_INCLUDES
)
set(POPART_DEPS_LIBS
)

set(POPART_INTERNAL_DEPS_INCLUDES
)
set(POPART_INTERNAL_DEPS_LIBS
  onnx
  onnx_proto
)

################################################################################
##### Sources
################################################################################
 
# core popart source files:
file(GLOB core_sources src/*cpp)
file(GLOB alias_sources src/alias/*cpp)
file(GLOB analysis_sources src/analysis/**/*cpp)
file(GLOB op_sources src/op/*cpp)
file(GLOB op_dyn_sources src/op/dynamic/*cpp)
file(GLOB op_exc_sources src/op/exchange/*cpp)
file(GLOB op_collectives_sources src/op/collectives/*cpp)
file(GLOB ces_sources src/ces/*cpp)
file(GLOB patterns_sources src/patterns/*cpp)
file(GLOB transforms_sources src/transforms/*cpp src/transforms/autodiff/*cpp)
file(GLOB subgraph_sources src/subgraph/*cpp)
file(GLOB onnxpasses_sources src/onnxpasses/*cpp)
file(GLOB nodepatterns_sources src/onnxpasses/nodepatterns/*cpp)

set(patterns_tgutil_sources
  src/patterns/tiedgatherutils/tgutils.cpp
  src/patterns/tiedgatherutils/tgutils.hpp
  src/patterns/tiedgatherutils/tgutils_impl.hpp
)

# poplar backend op specific:
file(GLOB x_sources src/popx/*cpp)
file(GLOB x_rng_sources src/popx/rng/*cpp)
file(GLOB opx_sources src/popx/op/*cpp)
file(GLOB opx_dyn_sources src/popx/op/dynamic/*cpp)
file(GLOB opx_exc_sources src/popx/op/exchange/*cpp)
file(GLOB opx_collectives_sources src/popx/op/collectives/*cpp)

set(sources
  ${core_sources}
  ${alias_sources}
  ${analysis_sources}
  ${op_sources}
  ${op_dyn_sources}
  ${op_exc_sources}
  ${op_collectives_sources}
  ${ces_sources}
  ${patterns_sources}
  ${patterns_tgutil_sources}
  ${transforms_sources}
  ${subgraph_sources}
  ${onnxpasses_sources}
  ${nodepatterns_sources}
)

# concatenate poplar backend sources to the list of files to compile
set(sources
  ${sources}
  ${x_sources}
  ${x_rng_sources}
  ${opx_sources}
  ${opx_dyn_sources}
  ${opx_exc_sources}
  ${opx_collectives_sources})

# Fed to version header.
execute_process(COMMAND
  git rev-parse --short=10 HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Provide fallback for building without a .git directory.
if(GIT_COMMIT_HASH STREQUAL "")
  set(GIT_COMMIT_HASH "0000000000")
endif()

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

    # Need position independent code to make dynamic library with static libs.
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(popart-only PRIVATE cxx_std_14)
target_compile_features(popart-only INTERFACE cxx_std_11)

# These definitions are private to popart-only, as the external user does not
# need them, but they are needed by all internal targets.
set(popart_common_definitions "")

# Hack: Poptorch doesn't need onnx, but it includes the monolithic names.hpp,
# which has many declarations intended to be private only that use
# ONNX_NAMESPACE. Thus, we make it a public definition in popart-only.
get_target_property(onnx_defs onnx INTERFACE_COMPILE_DEFINITIONS)
get_target_property(onnx_proto_defs onnx_proto INTERFACE_COMPILE_DEFINITIONS)
list(APPEND onnx_defs ${onnx_proto_defs})
list(REMOVE_DUPLICATES onnx_defs)

target_compile_definitions(popart-only
  PUBLIC
    ${onnx_defs}

  PRIVATE
    # Required to prevent warnings.
    PROTOBUF_INLINE_NOT_IN_HEADERS=0

    ${popart_common_definitions}
)

# These includes are private to popart-only, as the external user does not
# need them, but they are needed by all internal targets.
set(popart_common_includes
    # Include headers in src privately (not part of public API).
    ${PROJECT_SOURCE_DIR}/willow/src
)

target_include_directories(popart-only
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/willow/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>

    # For generated headers like version.hpp.in.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE
    ${popart_common_includes}
)

# Link dependencies, exposed at varying levels of visibility through interface
# targets. See [comment-willow-targets].

target_link_libraries(popart-only
  PRIVATE
    ${POPART_PRIVATE_DEPS_LIBS}
    ${POPART_DEPS_LIBS}
    ${POPART_INTERNAL_DEPS_LIBS}
  PUBLIC
    ${POPART_ONLY_DEPS_LIBS}
)
target_include_directories(popart-only SYSTEM
  PUBLIC
    ${POPART_ONLY_DEPS_INCLUDES}
)
target_include_directories(popart-only SYSTEM
  PRIVATE
    ${POPART_PRIVATE_DEPS_INCLUDES}
    ${POPART_DEPS_INCLUDES}
    ${POPART_INTERNAL_DEPS_INCLUDES}
)

add_coverage_flags_if_enabled(popart-only)

add_library(popart INTERFACE)
target_link_libraries(popart INTERFACE popart-only ${POPART_DEPS_LIBS})
target_include_directories(popart SYSTEM INTERFACE ${POPART_DEPS_INCLUDES})

add_library(popart-internal INTERFACE)
target_link_libraries(popart-internal INTERFACE popart ${POPART_INTERNAL_DEPS_LIBS})
target_include_directories(popart-internal SYSTEM INTERFACE ${POPART_INTERNAL_DEPS_INCLUDES})

target_include_directories(popart-internal INTERFACE ${popart_common_includes})
target_compile_definitions(popart-internal INTERFACE ${popart_common_definitions})

################################################################################
##### Install popart and popart-only
################################################################################

# Note, the GNUInstallDirs are cache variables that the user can override.
# We make the config install dir overridable too.
set(POPART_CMAKE_INSTALL_CONFIGDIR "${CMAKE_INSTALL_LIBDIR}/cmake/popart"
    CACHE PATH 
    "Install directory for Popart's CMake config files. Defaults to \
CMAKE_INSTALL_LIBDIR/cmake/popart, where CMAKE_INSTALL_LIBDIR is \
generated by the CMake package GNUInstallDirs."
)

# We install two separate components for popart and popart-only.
install(TARGETS popart-only
    EXPORT popart-only-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(TARGETS popart
    EXPORT popart-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT popart-only-targets
    FILE popart--popart-only-targets.cmake
    DESTINATION ${POPART_CMAKE_INSTALL_CONFIGDIR}
)
install(EXPORT popart-targets
    FILE popart--popart-targets.cmake
    DESTINATION ${POPART_CMAKE_INSTALL_CONFIGDIR}
)

install(DIRECTORY "include/popart"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "version.hpp.in" EXCLUDE
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/popart/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/popart
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

# For the user's convenience, we install our own Findgcl module, as gcl do not
# provide their own config file.
install(FILES "${PROJECT_SOURCE_DIR}/cmake/Modules/Findgcl.cmake"
        DESTINATION ${POPART_CMAKE_INSTALL_CONFIGDIR}/Modules
)

# See T6089: to use python3 where strictly required, and
# then remove this commented out code:
#> add_custom_target(
#>         python_ver_check
#>         COMMAND python -c "import sys; sys.exit(sys.version_info.major <= 2)"
#>         VERBATIM
#> )
#> add_dependencies(popart python_ver_check)
