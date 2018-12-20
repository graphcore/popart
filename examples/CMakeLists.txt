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

function(add_poponnx_py_example name)
  message(STATUS
    "Adding example '${name}'")
  add_custom_command(TARGET poponnx_run_examples
                     COMMAND bash ${CMAKE_BINARY_DIR}/examples/run_example.sh
		                  ${CMAKE_CURRENT_SOURCE_DIR}/python/${name}.py
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

add_poponnx_py_example(simple_addition)

set (CMAKE_CXX_STANDARD 11)

function(add_poponnx_cpp_example name)
  message(STATUS
    "Adding example '${name}' with sources '${ARGN}'")
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poponnx,INCLUDE_DIRECTORIES>)  
  target_link_libraries(${name} poponnx)
  set(executable_name "${name}")
  add_custom_command(TARGET poponnx_run_examples
                     COMMAND ${CMAKE_BINARY_DIR}/examples/${name}
		                 DEPENDS ${name} ${ARGN}
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})  
endfunction()


add_poponnx_cpp_example(custom_operator ${CMAKE_CURRENT_SOURCE_DIR}/cplusplus/custom_op.cpp)




install(DIRECTORY
  python
  DESTINATION ${INSTALL_EXAMPLES})

#install(DIRECTORY
#  cpp
#  DESTINATION ${INSTALL_EXAMPLES})

