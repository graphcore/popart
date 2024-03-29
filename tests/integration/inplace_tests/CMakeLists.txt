# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of inplace / view-changing logic
# to run only these tests with ctest, consider using something like,
# `ctest -R ip_test -j5 -VV'.

add_unit_test(accumulator_0_ip_test accumulator_0_ip_test.cpp VARIANTS "Hw")
add_unit_test(add_0_ip_test add_0_ip_test.cpp)
add_unit_test(addbias_0_ip_test addbias_0_ip_test.cpp)
add_unit_test(basic_0_ip_test basic_0_ip_test.cpp)
add_unit_test(concat_0_ip_test concat_0_ip_test.cpp)
add_unit_test(concat_1_ip_test concat_1_ip_test.cpp)
add_unit_test(concat_2_ip_test concat_2_ip_test.cpp)
add_unit_test(loop_0_ip_test loop_0_ip_test.cpp)
add_unit_test(mul_0_ip_test mul_0_ip_test.cpp)
add_unit_test(multi_nll_0_test multi_nll_0_test.cpp VARIANTS "IpuModel2")
add_unit_test(numerics_ip_nip_0_test numerics_ip_nip_0_test.cpp)
add_unit_test(numerics_ip_nip_1_test numerics_ip_nip_1_test.cpp)
add_unit_test(numerics_ip_with_outlining_0_test numerics_ip_with_outlining_0_test.cpp)
add_unit_test(pad_0_ip_test pad_0_ip_test.cpp)
add_unit_test(parallel_0_ip_test parallel_0_ip_test.cpp)
add_unit_test(pow_0_ip_test pow_0_ip_test.cpp)
add_unit_test(priorities_0_ip_test priorities_0_ip_test.cpp)
add_unit_test(scale_0_ip_test scale_0_ip_test.cpp)
add_unit_test(scale_1_ip_test scale_1_ip_test.cpp)
add_unit_test(series_0_ip_test series_0_ip_test.cpp)
add_unit_test(slice_0_ip_test slice_0_ip_test.cpp)
add_unit_test(slice_1_ip_test slice_1_ip_test.cpp)
add_unit_test(subsample_0_ip_test subsample_0_ip_test.cpp)
add_unit_test(anchor_0_ip_test anchor_0_ip_test.cpp)
add_unit_test(graph_output_0_ip_test graph_output_0_ip_test.cpp)
add_unit_test(restoreinplace_0_ip_test restoreinplace_0_ip_test.cpp VARIANTS "IpuModel2")

add_popart_py_unit_test(inplace_weight_test VARIANTS "IpuModel2")
