# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
from op_tester import op_tester


def test_gather_bigger(op_tester):
    d1 = np.random.rand(20, 68000, 20).astype(np.float32)
    d2 = np.array([0, 1]).astype(np.int32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.patterns = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')
