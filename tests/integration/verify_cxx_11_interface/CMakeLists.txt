# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Add a test is compiled with the c++11 standard to check that the popart
# interface is c++11 compatible. If not, the build will fail when compiling the
# test.
function(add_popart_cxx_11_unit_test name source)
  add_executable(${name} ${source})

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  target_link_libraries(${name} PRIVATE popart)

  # Because we use [[deprecated]] in our headers, which is C++14.
  # On Clang, with -pedantic,-Werror, this results in a compile error, so we
  # have to pass -Wno-c++14-extensions to disable the warning.
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    target_compile_options(${name} PRIVATE -Wno-c++14-extensions)
  endif()

  set_target_properties(
    ${name}
    PROPERTIES CXX_EXTENSIONS OFF
               CXX_STANDARD_REQUIRED ON
               CXX_STANDARD 11)

  add_test(
    NAME ${name}
    COMMAND ${name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# Bash script that generates a header that includes all popart public headers.
# Target depends on popart so is re-generated if popart changes.
add_custom_target(generate-all-headers ALL
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/generate_all_headers_hpp.sh" ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  BYPRODUCTS
    "${CMAKE_CURRENT_BINARY_DIR}/all_headers.hpp"
)
add_dependencies(generate-all-headers popart)

add_popart_cxx_11_unit_test(verify_cxx_11_interface verify_cxx_11_interface.cpp)
add_dependencies(verify_cxx_11_interface generate-all-headers)
