# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
# These unsqueeze tests are to verify the output of constexpr.
# To see that the const expr is actually being applied,
# see the tests in constexpr_tests directory
import numpy as np
from op_tester import op_tester


def test_slice_basic(op_tester):
    data = np.arange(2 * 3 * 4, dtype=np.float32)
    data = np.reshape(data, (2, 3, 4))

    dummy = np.zeros((2, 3, 1, 4, 1), dtype=np.float32)

    axes = [2, 4]

    def init_builder(builder):
        c = builder.aiOnnx.constant(data)
        u = builder.aiOnnx.unsqueeze([c], axes=axes)

        i1 = builder.addInputTensor(dummy)
        o = builder.aiOnnx.add([i1, u])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = data
        for axis in axes:
            result = np.expand_dims(result, axis=axis)
        return [result]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')
