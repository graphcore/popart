# Add a test is compiled with the c++11 standard to check that the popart
# interface is c++11 compatible. If not, the build will fail when compiling the
# test.
function(add_popart_cxx_11_unit_test name source)
  add_executable(${name} ${source})

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  target_link_libraries(${name} PRIVATE popart)

  set_target_properties(
    ${name}
    PROPERTIES CXX_EXTENSIONS OFF
               CXX_STANDARD_REQUIRED ON
               CXX_STANDARD 11)

  add_test(
    NAME ${name}
    COMMAND ${name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

# This test will include all headers in popart. Generate a file to include all
# headers for the test to use:
execute_process(
  COMMAND
    bash -c
    "echo \"$(for header in $(find ${PROJECT_SOURCE_DIR}/willow/include -name '*hpp' | sort);
                              do printf \"#include <$header>\n\";
                            done)\""
  OUTPUT_VARIABLE allHeaders)

# Write to build dir so as not to pollute the popart repo during build
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/all_headers.hpp" "${allHeaders}")

add_popart_cxx_11_unit_test(verify_cxx_11_interface verify_cxx_11_interface.cpp)
