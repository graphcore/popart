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

function(add_poponnx_unit_test name)
  message(STATUS
    "Adding test '${name}' with sources '${ARGN}'")
  add_test_executable(${name} ${ARGN})
  set(executable_name "${name}")
  add_test(NAME "${name}"
    COMMAND ${name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_poponnx_unit_test(sampletest sampletest.cpp)
add_poponnx_unit_test(dataflowtest dataflowtest.cpp)
add_poponnx_unit_test(earlyinfotest earlyinfotest.cpp)
add_poponnx_unit_test(matmultest matmul.cpp)
add_poponnx_unit_test(loggingtest loggingtest.cpp)
