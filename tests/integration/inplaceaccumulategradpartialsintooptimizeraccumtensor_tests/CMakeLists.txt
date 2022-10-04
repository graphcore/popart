# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

add_unit_test(
    inplaceaccumulategradpartialsintooptimizeraccumtensortests
    transform_unit_tests.cpp
    SUPPORT_LIBS test-graphs-test-util
)
