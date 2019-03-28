# for more useful warnings
include(EnableCompilerWarnings)

add_custom_target(poponnx_examples)
add_custom_command(TARGET poponnx_examples
  COMMAND cmake -E copy_directory
                   ${CMAKE_SOURCE_DIR}/examples/python
                   ${CMAKE_BINARY_DIR}/examples/python
)


add_custom_target(poponnx_run_examples
  DEPENDS poponnx_examples
)

add_custom_command(TARGET poponnx_run_examples
  COMMAND cmake -E copy
                   ${CMAKE_SOURCE_DIR}/examples/run_example.sh
                   ${CMAKE_BINARY_DIR}/examples/
)

## python examples
function(add_poponnx_py_example name)
  message(STATUS
    "Adding example '${name}'")
  add_custom_command(TARGET poponnx_run_examples
                     COMMAND bash ${CMAKE_BINARY_DIR}/examples/run_example.sh
		                  ${CMAKE_CURRENT_SOURCE_DIR}/python/${name}.py
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

function(add_poponnx_py_modelzoo_example name urlbase)
  message(STATUS
    "Adding example '${name}'")
  add_custom_command(TARGET poponnx_run_examples
                     COMMAND bash ${CMAKE_BINARY_DIR}/examples/run_example.sh
                      ${CMAKE_CURRENT_SOURCE_DIR}/python/model_zoo_import.py
                      ${urlbase}/${name}.tar.gz
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

add_poponnx_py_example(simple_addition)

set(urlbase "https://s3.amazonaws.com/download.onnx/models/opset_9")
#add_poponnx_py_modelzoo_example(bvlc_googlenet               ${urlbase})
#add_poponnx_py_modelzoo_example(bvlc_alexnet                 ${urlbase})
#add_poponnx_py_modelzoo_example(bvlc_reference_caffenet      ${urlbase})
#add_poponnx_py_modelzoo_example(bvlc_reference_rcnn_ilsvrc13 ${urlbase})
#add_poponnx_py_modelzoo_example(densenet121                  ${urlbase})
#add_poponnx_py_modelzoo_example(inception_v2                 ${urlbase})
#add_poponnx_py_modelzoo_example(resnet50                     ${urlbase})
#add_poponnx_py_modelzoo_example(shufflenet                   ${urlbase})
#add_poponnx_py_modelzoo_example(squeezenet                   ${urlbase})
#add_poponnx_py_modelzoo_example(vgg19                        ${urlbase})


## cpp examples
set (CMAKE_CXX_STANDARD 11)

function(add_poponnx_cpp_example name)
  message(STATUS
    "Adding example '${name}' with sources '${ARGN}'")
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${Boost_INCLUDE_DIRS}
      $<TARGET_PROPERTY:poponnx,INCLUDE_DIRECTORIES>)  
  target_link_libraries(${name}
    ${Boost_LIBRARIES}
    poponnx)
  set(executable_name "${name}")
  add_custom_command(TARGET poponnx_run_examples
                     COMMAND ${CMAKE_BINARY_DIR}/examples/${name}
		                 DEPENDS ${name} ${ARGN}
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})  
endfunction()


add_poponnx_cpp_example(custom_operator ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/custom_op.cpp)
add_poponnx_cpp_example(supported_operators ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/supported_ops.cpp)
add_poponnx_cpp_example(dot_inference_graph 
  ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/dot_inference_graph.cpp)

add_poponnx_cpp_example(dot_graph 
   ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/dot_graph.cpp)




install(DIRECTORY
  python
  DESTINATION ${INSTALL_EXAMPLES})

#install(DIRECTORY
#  cpp
#  DESTINATION ${INSTALL_EXAMPLES})

