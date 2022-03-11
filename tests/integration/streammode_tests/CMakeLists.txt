# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_py_unit_test(test_tensor_replication tensor_replication.py VARIANTS "Hw")
add_popart_py_unit_test(test_grouped_initialization grouped_initialization_test.py VARIANTS "Hw")
add_popart_py_unit_test(test_onnx_checkpointing grouped_initialization_test.py VARIANTS "Hw")