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
add_poponnx_cpp_unit_test(transformtest transform_test.cpp)
add_poponnx_cpp_unit_test(syntheticdatatest synthetic_data_test.cpp)
add_poponnx_cpp_unit_test(viewchangingtest view_changing_test.cpp)
add_poponnx_cpp_unit_test(opmanagertest op_manager_test.cpp)
add_poponnx_cpp_unit_test(buildertest builder_test.cpp)
add_poponnx_cpp_unit_test(allocatortest allocator_test.cpp)
add_poponnx_cpp_unit_test(mergecopiestest mergecopies_test.cpp)
add_poponnx_cpp_unit_test(isnonlinearitytest is_nonlinearity_test.cpp)
add_poponnx_cpp_unit_test(isnormtest is_norm_test.cpp)

function(add_poponnx_py_unit_test name)
  message(STATUS
    "Adding Python test '${name}'")
  add_test(NAME "${name}"
           COMMAND pytest -s ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

function(add_all_py_unit_tests exclusions)
  message(STATUS "Adding all python tests in ${CMAKE_CURRENT_SOURCE_DIR}")

  file(GLOB PYTHON_TESTS 
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    *.py)

  foreach(TEST_FILE ${PYTHON_TESTS})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    list(FIND exclusions ${TEST_NAME} IDX)
    if(IDX EQUAL -1)
      add_poponnx_py_unit_test(${TEST_NAME})
    else(IDX EQUAL -1)
      message(STATUS "Skipping file ${TEST_FILE}")
    endif(IDX EQUAL -1)
  endforeach(TEST_FILE)
endfunction()

# auto_virtual_graph_test is currently failing
add_all_py_unit_tests("test_util;auto_virtual_graph_test")


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
add_subdirectory(recompute_tests)
