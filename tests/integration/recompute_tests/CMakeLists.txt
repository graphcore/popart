add_popart_py_unit_test(implicit_recompute_regression_test implicit_recompute_regression_test.py)

#test(s) of Ir annotation (no device tests)
add_popart_cpp_unit_test(recompute_test_ir_none_annotation0 
                          recompute_test_ir_none_annotation0.cpp)

add_popart_cpp_unit_test(recompute_test_ir_normonly_annotation0 
                          recompute_test_ir_normonly_annotation0.cpp)

add_popart_cpp_unit_test(recompute_test_ir_standard_annotation0 
                          recompute_test_ir_standard_annotation0.cpp)

#test(s) of device calls to recompute annotated Ir
add_popart_cpp_unit_test(recompute_test_popx_normonly_calls0
                          recompute_test_popx_normonly_calls0.cpp)

add_popart_cpp_unit_test(recompute_test_popx_standard_calls0
                          recompute_test_popx_standard_calls0.cpp)
