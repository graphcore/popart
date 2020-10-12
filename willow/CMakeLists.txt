# for more useful warnings
include(EnableCompilerWarnings)

# Make a prefixed variable name for consistency (VERSION was defined earlier).
set(POPART_VERSION ${VERSION})

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
]]

################################################################################
##### Find dependencies
################################################################################

# Note, some of the depencies are "found" in the parent cmake file, some are
# found here.

find_package(Poplar REQUIRED)

# Search first for poprithms-static-config.cmake
find_package(poprithms-static CONFIG)
if (NOT TARGET poprithms-static)
  # For backwards compat fallback to FindPoprithms.cmake
  find_package(Poprithms REQUIRED)
endif ()

find_package(Spdlog REQUIRED)
message(STATUS "Spdlog include dir is " ${SPDLOG_INCLUDE_DIRS})

find_package(gcl REQUIRED)

################################################################################
##### Dependecy visibility levels (see [comment-willow-targets])
################################################################################

set(POPART_PRIVATE_DEPS_INCLUDES
    ${Boost_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
    # ${CBLAS_INCLUDE_DIRS}
    ${LIBPVTI_INCLUDE_DIR}
)
set(POPART_PRIVATE_DEPS_LIBS
    ${Boost_LIBRARIES}
    ${SPDLOG_LIBRARIES}
    # ${CBLAS_LIBRARIES}
    ${LIBPVTI_LIB}
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)

set(POPART_ONLY_DEPS_INCLUDES
)
set(POPART_ONLY_DEPS_LIBS
)

set(POPART_DEPS_INCLUDES
    ${POPLAR_INCLUDE_DIR}
    ${POPLIBS_INCLUDE_DIR}
    ${GCL_INCLUDE_DIR}
    # if the model checker is used: ${ONNX_UTILS_INCLUDE_DIR}
    ${ONNX_PB_INCLUDE_DIR}
    ${ONNX_CHECKER_INCLUDE_DIR}
    ${ONNX_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIR}
)
set(POPART_DEPS_LIBS
    ${POPLAR_LIB}
    ${POPLIN_LIB}
    ${POPNN_LIB}
    ${POPOPS_LIB}
    ${POPRAND_LIB}
    ${POPUTIL_LIB}
    ${GCL_LIBRARIES}
    # if the onnx mode checker is incorporated, this will be needed: ${ONNX_LIB}
    ${ONNX_LIB}
    ${ONNX_PROTO_LIB}
    ${PROTOBUF_LIB}
)
if (PoplarRunner_FOUND)
  list(APPEND POPART_DEPS_LIBS PoplarRunner::PoplarExecutableData)
endif()

set(POPART_INTERNAL_DEPS_INCLUDES
)
set(POPART_INTERNAL_DEPS_LIBS
)
if (TARGET poprithms-static)
  list(APPEND POPART_INTERNAL_DEPS_LIBS poprithms-static)
else()
  # Backwards compat for build_scripts using git tag before cmake exports of poprithms,
  # once this tag is incremented, we can delete FindPoprithms.cmake
  # and related ${POPRITHMS_*}, using normal cmake targets
  list(APPEND POPART_INTERNAL_DEPS_INCLUDES ${POPRITHMS_INCLUDE_DIR})
  list(APPEND POPART_INTERNAL_DEPS_LIBS ${POPRITHMS_LIB})
endif()


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

configure_file(
    include/popart/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/popart/version.hpp
    @ONLY
)

################################################################################
##### Build targets
################################################################################

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

# The following defines cause compiler warnings if not set
target_compile_definitions(popart-only
  PRIVATE 
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
)

target_include_directories(popart-only
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/willow/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>

    # For generated headers like version.hpp.in
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

# Link dependencies, exposed at varying levels of visibility through interface
# targets. See [comment-willow-targets].

target_link_libraries(popart-only
  PUBLIC
    ${POPART_ONLY_DEPS_LIBS}
  PRIVATE
    ${POPART_PRIVATE_DEPS_LIBS}
    ${POPART_DEPS_LIBS}
    ${POPART_INTERNAL_DEPS_LIBS}
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

add_library(popart INTERFACE)
target_link_libraries(popart INTERFACE popart-only ${POPART_DEPS_LIBS})
target_include_directories(popart SYSTEM INTERFACE ${POPART_DEPS_INCLUDES})

add_library(popart-internal INTERFACE)
target_link_libraries(popart-internal INTERFACE popart ${POPART_INTERNAL_DEPS_LIBS})
target_include_directories(popart-internal SYSTEM INTERFACE ${POPART_INTERNAL_DEPS_INCLUDES})

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
    "popart-config.cmake.in"
    "popart-config.cmake"
    INSTALL_DESTINATION ${POPART_CMAKE_INSTALL_CONFIGDIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR POPART_CMAKE_INSTALL_CONFIGDIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/popart-config-version.cmake
  VERSION ${POPART_VERSION}
  COMPATIBILITY SameMajorVersion
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
