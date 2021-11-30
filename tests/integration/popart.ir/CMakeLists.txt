# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_py_unit_test(popart.ir--test_dot_visualizer test_dot_visualizer.py VARIANTS IpuModel)
add_popart_py_unit_test(popart.ir--fwd_pipeline test_fwd_pipeline.py VARIANTS IpuModel)
add_popart_py_unit_test(popart.ir--milestone1-basic_subgraph test_milestone1_basic_subgraph.py VARIANTS IpuModel)
add_popart_py_unit_test(popart.ir--milestone2-autodiff test_milestone2_autodiff.py VARIANTS IpuModel)
add_popart_py_unit_test(popart.ir--modified_call test_modified_call.py VARIANTS IpuModel)
add_popart_py_unit_test(popart.ir--random-seed test_random_seed.py VARIANTS IpuModel)
add_popart_py_unit_test(popart.ir--remote_buffer test_remote_buffer.py VARIANTS Hw)
add_popart_py_unit_test(popart.ir--model_caching test_model_caching.py VARIANTS Hw)
