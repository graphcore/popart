# for more useful warnings
include(EnableCompilerWarnings)

include_directories("include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

configure_file(include/popart/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/popart/version.hpp
  @ONLY)

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

find_package(Poplar REQUIRED)

# concatenate poplar backend sources to the list of files to compile
set(sources
  ${sources}
  ${x_sources}
  ${opx_sources}
  ${opx_dyn_sources}
  ${opx_collectives_sources})

# Search first for poprithms-static-config.cmake
find_package(poprithms-static CONFIG)
if (NOT TARGET poprithms-static)
  # For backwards compat fallback to FindPoprithms.cmake
  find_package(Poprithms REQUIRED)
endif ()

find_package(Spdlog REQUIRED)
message(STATUS "Spdlog include dir is " ${SPDLOG_INCLUDE_DIRS})

find_package(gcl REQUIRED)

add_library(popart SHARED ${sources})

# Then need position indepenent code enabled to make dyanamic library with static libs
set_property(TARGET popart PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories (popart SYSTEM PUBLIC
    # if the model checker is used: ${ONNX_UTILS_INCLUDE_DIR}
    ${ONNX_PB_INCLUDE_DIR}
    ${ONNX_CHECKER_INCLUDE_DIR}
    ${ONNX_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
    ${CBLAS_INCLUDE_DIRS}
    ${POPLAR_INCLUDE_DIR}
    ${POPLIBS_INCLUDE_DIR}
    ${GCL_INCLUDE_DIR}
    )

target_link_libraries(popart
  # if the onnx mode checker is incorporated, this will be needed: ${ONNX_LIB}
  ${ONNX_LIB}
  ${ONNX_PROTO_LIB}
  ${PROTOBUF_LIB}
  ${CBLAS_LIBRARIES}
  ${Boost_LIBRARIES}
  ${SPDLOG_LIBRARIES}
  ${POPLAR_LIB}
  ${POPLIN_LIB}
  ${POPNN_LIB}
  ${POPOPS_LIB}
  ${POPRAND_LIB}
  ${POPUTIL_LIB}
  ${GCL_LIBRARIES}
  ${CMAKE_DL_LIBS}
  ${CMAKE_THREAD_LIBS_INIT}
)

if (PoplarRunner_FOUND)
  target_link_libraries(popart
    PoplarRunner::PoplarExecutableData
)
endif()

if (TARGET poprithms-static)
  target_link_libraries(popart poprithms-static)
else()
  # Backwards compat for build_scripts using git tag before cmake exports of poprithms,
  # once this tag is incremented, we can delete FindPoprithms.cmake
  # and related ${POPRITHMS_*}, using normal cmake targets
  target_include_directories(popart SYSTEM PUBLIC ${POPRITHMS_INCLUDE_DIR})
  target_link_libraries(popart ${POPRITHMS_LIB})
endif()

# The following defines cause compiler warnings if not set
target_compile_definitions(popart
  PRIVATE PROTOBUF_INLINE_NOT_IN_HEADERS=0
)


install(TARGETS popart
        EXPORT popart
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popart)

install(DIRECTORY "include/popart"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popart)

# See T6089: to use python3 where strictly required, and
# then remove this commented out code:
#> add_custom_target(
#>         python_ver_check
#>         COMMAND python -c "import sys; sys.exit(sys.version_info.major <= 2)"
#>         VERBATIM
#> )
#> add_dependencies(popart python_ver_check)
