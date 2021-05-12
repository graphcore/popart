# Tests of inplace / view-changing logic
# to run only these tests with ctest, consider using something like,
# `ctest -R ip_test -j5 -VV'.

add_popart_cpp_unit_test(schedule_liveness_0_test.cpp schedule_liveness_0_test.cpp)

add_popart_py_unit_test(serialization_to_poprithms_shift_graph_test0)
