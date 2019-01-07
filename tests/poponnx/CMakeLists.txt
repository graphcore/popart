find_package(Boost REQUIRED unit_test_framework)

function(add_test_executable name)
    add_executable(${name} ${ARGN})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poponnx,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name}
    poponnx
    Boost::unit_test_framework
  )
  if (NOT Boost_UNIT_TEST_FRAMEWORK_LIBRARY MATCHES "\\.a$")
    target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  endif()
endfunction()

function(add_poponnx_cpp_unit_test name)
  message(STATUS
    "Adding C++ test '${name}' with sources '${ARGN}'")
  add_test_executable(${name} ${ARGN})
  set(executable_name "${name}")
  add_test(NAME "${name}"
           COMMAND ${name}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_poponnx_cpp_unit_test(dataflowtest dataflowtest.cpp)
add_poponnx_cpp_unit_test(inputshapeinfotest inputshapeinfotest.cpp)
add_poponnx_cpp_unit_test(loggingtest loggingtest.cpp)
add_poponnx_cpp_unit_test(exceptiontest exceptiontest.cpp)
add_poponnx_cpp_unit_test(matmultest matmul.cpp)
add_poponnx_cpp_unit_test(numpybroadcastshapetest numpybroadcastshapetest.cpp)
add_poponnx_cpp_unit_test(patternstest patterns_test.cpp)
add_poponnx_cpp_unit_test(prunetest prune_test.cpp)
add_poponnx_cpp_unit_test(recomputetest recompute_test.cpp)
add_poponnx_cpp_unit_test(transformtest transform_test.cpp)
add_poponnx_cpp_unit_test(syntheticdatatest synthetic_data_test.cpp)
add_poponnx_cpp_unit_test(viewchangingtest view_changing_test.cpp)
add_poponnx_cpp_unit_test(constexprtest constexpr_test.cpp)
add_poponnx_cpp_unit_test(opmanagertest op_manager_test.cpp)

function(add_poponnx_py_unit_test name)
  message(STATUS
    "Adding Python test '${name}'")
  add_test(NAME "${name}"
           COMMAND pytest -s ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_poponnx_py_unit_test(builder_test)
add_poponnx_py_unit_test(graph_caching_test)
add_poponnx_py_unit_test(loader_test)
add_poponnx_py_unit_test(net_test)
add_poponnx_py_unit_test(operators_test)
add_poponnx_py_unit_test(anchorreturntype_test)
add_poponnx_py_unit_test(options_test)
add_poponnx_py_unit_test(report_test)
add_poponnx_py_unit_test(patterns_test)
add_poponnx_py_unit_test(fp16_test)
add_poponnx_py_unit_test(variable_inference_test)
add_poponnx_py_unit_test(virtual_graph_check_test)
add_poponnx_py_unit_test(virtual_graph_test)
add_poponnx_py_unit_test(optimizer_test)
add_poponnx_py_unit_test(export_test)
