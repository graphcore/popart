# for more useful warnings
include(EnableCompilerWarnings)

include_directories("../neuralnet/include")

find_package(pybind11 REQUIRED)

# https://pybind11.readthedocs.io/en/master/faq.html
# suggests not using find_package(PythonLibs), because 
# pybind11 finds Python itself
message(STATUS "Python include dir: " ${PYTHON_INCLUDE_DIR})
message(STATUS "Python library : " ${PYTHON_LIBRARY})
# find_package(PythonLibs)

pybind11_add_module(pyneuralnet pyneuralnet.cpp)

target_include_directories (pyneuralnet PUBLIC 
    # if the model checker is used: ${ONNX_UTILS_INCLUDE_DIR}
    ${ONNX_PB_INCLUDE_DIR}
    ${ONNX_CHECKER_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIR}
    ${BOOST_FILESYSTEM_INCLUDE_DIR}
    ${CBLAS_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    )

  target_link_libraries(pyneuralnet PRIVATE
  neuralnet
  ${ONNX_LIB}
  ${CBLAS_LIBRARIES}
  ${ONNX_PROTO_LIB}
  ${Protobuf_LIBRARIES} 
  ${BOOST_FILESYSTEM_LIB}
  ${BOOST_SYSTEM_LIB}
  ${PYTHON_LIBRARY}
)

install(TARGETS pyneuralnet
  # destination : this directory
  DESTINATION ${PYNEURALNET_DIR})

