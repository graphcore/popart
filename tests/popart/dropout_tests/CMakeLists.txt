add_popart_cpp_unit_test(dropout_on_weight_test_0
                         dropout_on_weight_test_0.cpp VARIANTS "IpuModel;Hw")

#TODO(T25465): this test may not be needed once dropout is outlineable
add_popart_cpp_unit_test(dropout_basic dropout_basic.cpp)