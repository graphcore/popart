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
add_poponnx_cpp_unit_test(numpybroadcastshapetest numpybroadcastshapetest.cpp)
add_poponnx_cpp_unit_test(prunetest prune_test.cpp)
add_poponnx_cpp_unit_test(recomputetest recompute_test.cpp)
add_poponnx_cpp_unit_test(transformtest transform_test.cpp)
add_poponnx_cpp_unit_test(syntheticdatatest synthetic_data_test.cpp)
add_poponnx_cpp_unit_test(viewchangingtest view_changing_test.cpp)
add_poponnx_cpp_unit_test(opmanagertest op_manager_test.cpp)
add_poponnx_cpp_unit_test(buildertest builder_test.cpp)
add_poponnx_cpp_unit_test(allocatortest allocator_test.cpp)
add_poponnx_cpp_unit_test(mergecopiestest mergecopies_test.cpp)

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
add_poponnx_py_unit_test(anchorreturntype_test)
add_poponnx_py_unit_test(options_test)
add_poponnx_py_unit_test(report_test)
add_poponnx_py_unit_test(patterns_test)
add_poponnx_py_unit_test(padsumpattern)
add_poponnx_py_unit_test(fp16_test)
add_poponnx_py_unit_test(variable_inference_test)
add_poponnx_py_unit_test(virtual_graph_check_test)
add_poponnx_py_unit_test(virtual_graph_test)
add_poponnx_py_unit_test(optimizer_test)
add_poponnx_py_unit_test(export_test)
add_poponnx_py_unit_test(ipu_copy_test)
add_poponnx_py_unit_test(float_to_half_conversion_test)
add_poponnx_py_unit_test(recompute_test)
add_poponnx_py_unit_test(ipu_gather_test)
add_poponnx_py_unit_test(loss_test)
add_poponnx_py_unit_test(all_constexpr)
add_poponnx_py_unit_test(graph_replication_test)
add_poponnx_py_unit_test(saved_executable)
add_poponnx_py_unit_test(train_then_infer_test)

add_subdirectory(operators_test)
add_subdirectory(constexpr_tests)
add_subdirectory(inplace_tests)
add_subdirectory(dot_tests)
add_subdirectory(transformation_tests)
add_subdirectory(pattern_tests)
add_subdirectory(subgraph_tests)
add_subdirectory(logical_if_tests)
add_subdirectory(topk_tests)
add_subdirectory(matmul_tests)
add_subdirectory(session_api_tests)
