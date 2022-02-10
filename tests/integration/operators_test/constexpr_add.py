# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np


def test_add_basic(op_tester):
    d1 = np.asarray([1, 2, 3], dtype=np.float32)
    d2 = np.asarray([4, 5, 6], dtype=np.float32)
    dummy = np.asarray([0, 0, 0], dtype=np.float32)

    def init_builder(builder):
        c1 = builder.aiOnnx.constant(d1)
        c2 = builder.aiOnnx.constant(d2)
        cadd = builder.aiOnnx.add([c1, c2])

        i1 = builder.addInputTensor(dummy)
        o = builder.aiOnnx.add([i1, cadd])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = d1 + d2 + dummy
        return [result]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')
