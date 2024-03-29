# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest


def test_matmul_scalar_input(op_tester):
    """
    Test the exception raised when an intput to a matmul is a scalar
    """

    d1 = np.array((2), dtype=np.float32)
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

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference)

    assert (
        e_info.value.args[0]
        == "Op(ai.onnx.MatMul:9, inputs=[input, input/1], outputs=[MatMul:0]) doesn't support scalar tensor input as the lhs input"
    )
