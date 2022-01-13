# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

set(test_util_root_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(test_util_root_public_headers_dir ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil)

set(test_util_include_dir_for_libs ${CMAKE_CURRENT_SOURCE_DIR}/include)

############# test_graphs #############

set(test_graphs_test_util_src_dir ${test_util_root_src_dir}/test_graphs)
set(test_graphs_test_util_public_headers_dir
    ${test_util_root_public_headers_dir}/test_graphs
)

set(test_graphs_test_util_sources
    ${test_graphs_test_util_src_dir}/builder.cpp
    ${test_graphs_test_util_src_dir}/dummy_builder.cpp
    ${test_graphs_test_util_src_dir}/graph_test_models.cpp
    ${test_graphs_test_util_src_dir}/ready_made.cpp
    ${test_graphs_test_util_src_dir}/op/dummy.cpp
)

set(test_graphs_test_util_public_headers
    ${test_graphs_test_util_public_headers_dir}/builder.hpp
    ${test_graphs_test_util_public_headers_dir}/dummy_builder.hpp
    ${test_graphs_test_util_public_headers_dir}/graph_test_models.hpp
    ${test_graphs_test_util_public_headers_dir}/ready_made.hpp
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
    PUBLIC popart-internal
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
      CXX_EXTENSIONS OFF

      # Need pic to make dynamic library with static libs.
      POSITION_INDEPENDENT_CODE ON
)
target_compile_features(test-graphs-test-util PUBLIC cxx_std_14)

############# irquery #############

set(irquery_test_util_src_dir ${test_util_root_src_dir}/irquery)
set(irquery_test_util_public_headers_dir
    ${test_util_root_public_headers_dir}/irquery
)

set(irquery_test_util_sources
    ${irquery_test_util_src_dir}/graphtestwrapper.cpp
    ${irquery_test_util_src_dir}/irtestwrapper.cpp
    ${irquery_test_util_src_dir}/opstestwrapper.cpp
    ${irquery_test_util_src_dir}/tensorindexmaptestwrapper.cpp
    ${irquery_test_util_src_dir}/tensorindextestwrapper.cpp
    ${irquery_test_util_src_dir}/tensortestwrapper.cpp
    ${irquery_test_util_src_dir}/testfailuretriggerer.cpp
)

set(irquery_test_util_public_headers
    ${irquery_test_util_public_headers_dir}/irquery.hpp
    ${irquery_test_util_public_headers_dir}/require.hpp
    ${irquery_test_util_public_headers_dir}/graphtestwrapper.hpp
    ${irquery_test_util_public_headers_dir}/irtestwrapper.hpp
    ${irquery_test_util_public_headers_dir}/opstestwrapper.hpp
    ${irquery_test_util_public_headers_dir}/optestwrapper.hpp
    ${irquery_test_util_public_headers_dir}/tensorindexmaptestwrapper.hpp
    ${irquery_test_util_public_headers_dir}/tensorindextestwrapper.hpp
    ${irquery_test_util_public_headers_dir}/tensortestwrapper.hpp
    ${irquery_test_util_public_headers_dir}/testfailuretriggerer.hpp
    ${irquery_test_util_public_headers_dir}/testwrapper.hpp
)

add_library(ir-query-test-util SHARED
    ${irquery_test_util_sources}
    ${irquery_test_util_public_headers}
)
target_include_directories(ir-query-test-util
    PUBLIC
      ${test_util_include_dir_for_libs}
)
target_link_libraries(ir-query-test-util
    PUBLIC popart-internal
)
link_boost(ir-query-test-util)

set_target_properties(ir-query-test-util
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
      CXX_EXTENSIONS OFF

      # Need pic to make dynamic library with static libs.
      POSITION_INDEPENDENT_CODE ON
)
target_compile_features(ir-query-test-util PUBLIC cxx_std_14)
