# Tests of visualization file logic 
# to run only these tests with ctest, consider using something like,
# `ctest -R _dot_ -j5 -VV'.

# to generate the pdfs and visualize them, something like,
# dot -Tpdf tests/popart/dot_tests/fwd0.dot -o tests/popart/dot_tests/fwd0.pdf

add_popart_cpp_unit_test(basic_0_dot_test basic_0_dot_test.cpp)
add_popart_cpp_unit_test(nested_0_dot_test nested_0_dot_test.cpp)

add_popart_cpp_unit_test(dot_inference_graph_test dot_inference_graph_test.cpp)

add_popart_py_unit_test(dot_env_test)
