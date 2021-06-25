# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

add_popart_python_module(popart_internal_ir
    popart._internal.ir.cpp
    bindings/graph.cpp
    bindings/graphid.cpp
    bindings/ir.cpp
)

target_include_directories(popart_internal_ir PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

configure_file(__init__.py.in
    "${CMAKE_CURRENT_BINARY_DIR}/popart/_internal/ir/__init__.py"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/popart/_internal/ir/__init__.py"
    DESTINATION "${INSTALL_PYDIR}/popart/_internal/ir/"
)
