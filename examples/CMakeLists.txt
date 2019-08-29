
# for more useful warnings
include(EnableCompilerWarnings)


add_custom_target(popart_examples)
add_custom_command(TARGET popart_examples
  COMMAND cmake -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/python
                   ${CMAKE_CURRENT_BINARY_DIR}/python
)

# Required for including examples in documentation
add_custom_command(TARGET popart_examples
  COMMAND cmake -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus
                   ${CMAKE_CURRENT_BINARY_DIR}/cplusplus
)


add_custom_target(popart_run_examples
  DEPENDS popart_examples
)

add_custom_command(TARGET popart_run_examples
  COMMAND cmake -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/run_example.sh
                   ${CMAKE_CURRENT_BINARY_DIR}/
)

## python examples
function(add_popart_py_example name)
  message(STATUS
    "Adding example '${name}'")
  add_custom_command(TARGET popart_run_examples
                     COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/run_example.sh
		                 ${CMAKE_CURRENT_SOURCE_DIR}/python/${name}.py
                     WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endfunction()

function(add_popart_py_modelzoo_example name urlbase)
  message(STATUS
    "Adding example '${name}'")
  add_custom_command(TARGET popart_run_examples
                     COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/run_example.sh
                     ${CMAKE_CURRENT_SOURCE_DIR}/python/model_zoo_import.py
                     ${urlbase}/${name}.tar.gz
                     WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endfunction()

add_popart_py_example(simple_addition)

set(urlbase "https://s3.amazonaws.com/download.onnx/models/opset_9")
#add_popart_py_modelzoo_example(bvlc_googlenet               ${urlbase})
#add_popart_py_modelzoo_example(bvlc_alexnet                 ${urlbase})
#add_popart_py_modelzoo_example(bvlc_reference_caffenet      ${urlbase})
#add_popart_py_modelzoo_example(bvlc_reference_rcnn_ilsvrc13 ${urlbase})
#add_popart_py_modelzoo_example(densenet121                  ${urlbase})
#add_popart_py_modelzoo_example(inception_v2                 ${urlbase})
#add_popart_py_modelzoo_example(resnet50                     ${urlbase})
#add_popart_py_modelzoo_example(shufflenet                   ${urlbase})
#add_popart_py_modelzoo_example(squeezenet                   ${urlbase})
#add_popart_py_modelzoo_example(vgg19                        ${urlbase})


## cpp examples
set (CMAKE_CXX_STANDARD 14)

function(add_popart_cpp_example name)
  message(STATUS
    "Adding example '${name}' with sources '${ARGN}'")
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${Boost_INCLUDE_DIRS}
      $<TARGET_PROPERTY:popart,INCLUDE_DIRECTORIES>)  
  target_link_libraries(${name}
    ${Boost_LIBRARIES}
    popart)
  set(executable_name "${name}")
  add_custom_command(TARGET popart_run_examples
                     COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${name}
		                 DEPENDS ${name} ${ARGN}
                     WORKING_DIRECTORY ${PROJECT_BINARY_DIR})  
endfunction()


add_popart_cpp_example(custom_operator ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/custom_op.cpp)
add_popart_cpp_example(supported_operators ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/supported_ops.cpp)
add_popart_cpp_example(dot_inference_graph 
  ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/dot_inference_graph.cpp)

add_popart_cpp_example(dot_graph 
   ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/dot_graph.cpp)

install(DIRECTORY
  python
  DESTINATION ${INSTALL_EXAMPLES})

install(DIRECTORY
  cplusplus
  DESTINATION ${INSTALL_EXAMPLES})

