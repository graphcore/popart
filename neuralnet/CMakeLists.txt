# for more useful warnings
include(EnableCompilerWarnings)

add_definitions(-std=c++11)
include_directories("include")
file(GLOB_RECURSE source_files src/*cpp)
add_library(neuralnet SHARED ${source_files})

MESSAGE( STATUS "BOOST_FILESYSTEM_LIB to be used : " ${BOOST_FILESYSTEM_LIB} )
MESSAGE( STATUS "BOOST_FILESYSTEM_INCLUDE_DIR to be used :  " ${BOOST_FILESYSTEM_INCLUDE_DIR} )

target_include_directories (neuralnet PUBLIC 
    # if the model checker is used: ${ONNX_UTILS_INCLUDE_DIR}
    ${ONNX_PB_INCLUDE_DIR}
    ${ONNX_CHECKER_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIR}
    ${BOOST_FILESYSTEM_INCLUDE_DIR}
    ${CBLAS_INCLUDE_DIRS}
    )

target_link_libraries(neuralnet
  # if the onnx mode checker is incorporated, this will be needed: ${ONNX_LIB}
  ${ONNX_LIB}
  ${CBLAS_LIBRARIES}
  ${ONNX_PROTO_LIB}
  ${Protobuf_LIBRARIES} 
  ${BOOST_FILESYSTEM_LIB}
  # filesystem (which is used) requires system
  ${BOOST_SYSTEM_LIB}
)


