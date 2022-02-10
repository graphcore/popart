# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np


def test_gather_big(op_tester):
    d1 = np.random.rand(20, 48000, 20).astype(np.float32)
    d2 = np.array([0, 1]).astype(np.int32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    # Opx alias/modify testing takes too long on large gathers
    op_tester.options.opxAliasChecking = False
    op_tester.options.opxModifyChecking = False
    op_tester.run(init_builder, reference, 'infer')
