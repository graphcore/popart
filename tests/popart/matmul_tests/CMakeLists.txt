# Tests of ONNX matmul

add_popart_cpp_unit_test(basic0_matmul_test 
                         basic0_matmul_test.cpp)

add_popart_cpp_unit_test(dataless0_train_matmul_test 
                          dataless0_train_matmul_test.cpp)

add_popart_cpp_unit_test(matmulgradpattern_test 
                         matmulgradpattern_test.cpp)
