# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of session API robustness to user errors
# to run only these tests with ctest, consider using something like,
# `ctest -R session_api -j5 -VV'.

add_unit_test(basic_0_session_api_test basic_0_session_api_test.cpp)
add_unit_test(get_tensor_ids_test get_tensor_ids_test.cpp)

add_popart_py_unit_test(reset_host_weights_test)

add_popart_py_unit_test(rng_test VARIANTS "${HW_EXCLUDE_C600}")
