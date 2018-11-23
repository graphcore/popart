include(GNUInstallDirs)

# for more useful warnings
include(EnableCompilerWarnings)

option(POPLAR_BACKEND "Build Poplar backend (Graphcore IPU)" ON)
include_directories("include")


file(GLOB sources src/*cpp)
file(GLOB op_sources src/op/*cpp)
file(GLOB opx_sources src/popx/op/*cpp)

if (POPLAR_BACKEND)
message( STATUS "Building WITH Poplar IPU backend")
find_package(Poplar REQUIRED)
# concatenate poplar backend sources to the list of files to compile
file(GLOB pop_backend_sources src/popx/*cpp)
set(sources ${sources} ${pop_backend_sources})
else()
message( STATUS "Building WITHOUT Poplar IPU backend")
endif()

find_package(Spdlog REQUIRED)
message(STATUS "spdlog at " ${Spdlog_INCLUDE_DIRS})

add_library(poponnx SHARED ${sources} ${op_sources} ${opx_sources})

target_include_directories (poponnx SYSTEM PUBLIC
    # if the model checker is used: ${ONNX_UTILS_INCLUDE_DIR}
    ${ONNX_PB_INCLUDE_DIR}
    ${ONNX_CHECKER_INCLUDE_DIR}
    ${ONNX_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
	${SPDLOG_INCLUDE_DIRS}
    ${CBLAS_INCLUDE_DIRS}
    ${POPLAR_INCLUDE_DIR}
    ${POPLIBS_INCLUDE_DIR}
    )

target_link_libraries(poponnx
  # if the onnx mode checker is incorporated, this will be needed: ${ONNX_LIB}
  ${ONNX_LIB}
  ${ONNX_PROTO_LIB}
  ${Protobuf_LIBRARIES}
  ${CBLAS_LIBRARIES}
  ${Boost_LIBRARIES}
  ${SPDLOG_LIBRARIES}
  ${POPLAR_LIB}
  ${POPLIN_LIB}
  ${POPNN_LIB}
  ${POPOPS_LIB}
  ${POPUTIL_LIB}
)

# The following defines cause compiler warnings if not set
target_compile_definitions(poponnx
  PRIVATE PROTOBUF_INLINE_NOT_IN_HEADERS=0
)


install(TARGETS poponnx
        EXPORT poponnx
	      DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poponnx
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
