# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
add_library(attr_pylist_float_customOp SHARED attr_pylist_float_customOp.cpp)

target_link_libraries(attr_pylist_float_customOp
    popart-internal
)

install(TARGETS attr_pylist_float_customOp DESTINATION ${INSTALL_TESTS}/integration/customop_attr_pylist_tests)

add_popart_py_unit_test(customop_attr_pylist_float_test)
