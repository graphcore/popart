# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


def test_matmul_mismatched_inputs(op_tester):
    """
    Test the exception raised when the inputs to a matmul are mismatched
    """

    d1 = np.random.rand(3, 4).astype(np.float32)
    d2 = np.random.rand(3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference)

    assert (
        e_info.value.args[0] ==
        "Op(ai.onnx.MatMul:9, inputs=[input, input/1], outputs=[MatMul:0]) contracting dimensions unequal: lhs 'input' [3 4], rhs 'input/1' [3 4]"
    )
