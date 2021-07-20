# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

add_popart_python_module(popart_internal_ir
    ../popart/shared_cpp/np_utils.cpp
    bindings/debugcontext.cpp
    bindings/graph.cpp
    bindings/graphid.cpp
    bindings/ir.cpp
    bindings/scope.cpp
    bindings/tensor.cpp
    bindings/tensordata.cpp
    bindings/tensorinfo.cpp
    popart._internal.ir.cpp
    bindings/tensors.cpp
    ../popart/shared_cpp/np_utils.cpp
)

target_include_directories(popart_internal_ir PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# The top-level "__init__.py" file for popart is in the main package directory
# "${CMAKE_CURRENT_BINARY_DIR}/../popart/popart/". The
# "_internal/ir/__init__.py" file should be in a child dir of the main dir for
# scripts, such as "docs/gen_supported_ops.py" to work, since they are executed
# from the build dir.
configure_file(__init__.py.in
    "${CMAKE_CURRENT_BINARY_DIR}/../popart/popart/_internal/ir/__init__.py"
    @ONLY
)

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
    DESTINATION "${INSTALL_PYDIR}/popart/_internal/ir/"
    FILES_MATCHING PATTERN "*.pyi"
)
