# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

set(test_device_root_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(test_device_root_public_headers_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(
  testdevice OBJECT "${test_device_root_src_dir}/testdevice.cpp"
                    "${test_device_root_public_headers_dir}/testdevice.hpp")
target_link_libraries(testdevice PUBLIC popart-internal Boost::boost)

target_include_directories(testdevice
                           PUBLIC ${test_device_root_public_headers_dir})

target_compile_features(testdevice PRIVATE cxx_std_17)
