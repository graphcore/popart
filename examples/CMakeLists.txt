# for more useful warnings
include(EnableCompilerWarnings)

add_custom_target(poponnx_examples)
add_custom_command(TARGET poponnx_examples
  COMMAND cmake -E copy_directory
                   ${CMAKE_SOURCE_DIR}/examples/python
                   ${CMAKE_BINARY_DIR}/examples/python
)
add_custom_command(TARGET poponnx_examples
  COMMAND cmake -E copy
                   ${CMAKE_SOURCE_DIR}/examples/run_example.sh
                   ${CMAKE_BINARY_DIR}/examples/
)

add_custom_target(poponnx_examples)

function(add_poponnx_py_example name)
  message(STATUS
    "Adding example '${name}'")
  add_custom_command(TARGET poponnx_examples
                     COMMAND bash ${CMAKE_BINARY_DIR}/examples/run_example.sh
		                  ${CMAKE_CURRENT_SOURCE_DIR}/python/${name}.py
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

add_poponnx_py_example(simple_addition)


install(DIRECTORY
  python
  DESTINATION ${INSTALL_EXAMPLES})

install(DIRECTORY
  cpp
  DESTINATION ${INSTALL_EXAMPLES})

