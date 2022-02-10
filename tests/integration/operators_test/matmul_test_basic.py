# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np


def test_matmul_basic(op_tester):
    d1 = np.random.rand(2, 3).astype(np.float32)
    d2 = np.random.rand(3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)
