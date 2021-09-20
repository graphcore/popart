# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_py_unit_test(unittest_python_popart.ir_dtype test_dtype.py)
add_popart_py_unit_test(unittest_python_popart.ir_graph test_graph.py)
add_popart_py_unit_test(unittest_python_popart.ir_ir test_ir.py)
add_popart_py_unit_test(unittest_python_popart.ir_tensor test_tensor.py)
add_popart_py_unit_test(unittest_python_popart.ir_ops_add ops/test_add.py)
add_popart_py_unit_test(unittest_python_popart.ir_ops_gather ops/test_gather.py)
