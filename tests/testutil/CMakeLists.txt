# For all testutil libs.
include(EnableCompilerWarnings)

set(test_util_root_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(test_util_root_public_headers_dir ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil)

set(test_util_include_dir_for_libs ${CMAKE_CURRENT_SOURCE_DIR}/include)

############# test_graphs #############

set(test_graphs_test_util_src_dir ${test_util_root_src_dir}/test_graphs)
set(test_graphs_test_util_public_headers_dir
    ${test_util_root_public_headers_dir}/test_graphs
)

set(test_graphs_test_util_sources
    ${test_graphs_test_util_src_dir}/graphs.cpp
    ${test_graphs_test_util_src_dir}/op/dummy.cpp
)

set(test_graphs_test_util_public_headers
    ${test_graphs_test_util_public_headers_dir}/graphs.hpp
    ${test_graphs_test_util_public_headers_dir}/op/dummy.hpp
)

add_library(test-graphs-test-util SHARED
    ${test_graphs_test_util_sources}
    ${test_graphs_test_util_public_headers}
)
target_include_directories(test-graphs-test-util
    PUBLIC
      ${test_util_include_dir_for_libs}
)
target_link_libraries(test-graphs-test-util
    PRIVATE popart
)
set_target_properties(test-graphs-test-util
    PROPERTIES
      # Must use C++14 as popart is only C++14 and has headers that define
      # the same things differently depending on the C++ standard, e.g.
      # optional.hpp. Thus, using C++17 will result in undefined symbols at link
      # time.
      #
      # This is because, when libpopart.so was compiled, which has the
      # implementation of the function, the symbols will have certain names
      # based on the defines in the header.
      # 
      # When we include those same headers under C++17, the defines are
      # different, so different symbol names are generated. This then results in
      # undefined references at link time.
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO

      # Need pic to make dynamic library with static libs.
      POSITION_INDEPENDENT_CODE ON
)
