# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_python_module(popart_internal_ir popart._internal.ir.cpp)

configure_file(__init__.py.in
    "${CMAKE_CURRENT_BINARY_DIR}/popart/_internal/ir/__init__.py"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/popart/_internal/ir/__init__.py"
    DESTINATION "${INSTALL_PYDIR}/popart/_internal/ir/"
)