# TODO: T17855 figure out why this is failing on hardware.
add_popart_cpp_unit_test(dropout_on_weight_test_0
                          dropout_on_weight_test_0.cpp VARIANTS "IpuModel")
