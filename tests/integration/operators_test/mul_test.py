# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent)


def test_mul(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.mul([i1, i2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 * t2

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, t1.grad, t2.grad, None]

    op_tester.setPatterns(["PreUniRepl", "MulArgGradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="train")


def test_broadcast_mul(op_tester):
    d1 = np.random.rand(2, 2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.mul([i1, i2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 * t2

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, t1.grad, t2.grad, None]

    op_tester.setPatterns(["PreUniRepl", "MulArgGradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="train")


input_infos = (
    ([], np.float16, [], np.float32),
    ([2, 1], np.float16, [], np.float32),
    ([2, 1], np.float16, [1], np.float32),
    ([1], np.float32, [2, 2], np.float16),
)


@pytest.mark.parametrize("in_infos", input_infos)
def test_mixed_precision_floating_point_mul(in_infos, op_tester):
    (shape0, type0, shape1, type1) = in_infos
    d1 = np.array(np.random.rand(*shape0)).astype(type0)
    d2 = np.array(np.random.rand(*shape1)).astype(type1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.mul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1).float()
        t2 = torch.tensor(d2).float()
        out = t1 * t2
        # poplar takes fp16 output type in case of mixed precision inputs
        return [out.half()]

    op_tester.atol = 1e-03
    op_tester.run(init_builder, reference)


def test_fp16_and_nonscalar_fp32_input_mul(op_tester):
    d1 = np.array(np.random.rand(2, 2).astype(np.float16))
    d2 = np.array(np.random.rand(2, 2).astype(np.float32))

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.mul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return [None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference)
    assert e_info.value.args[0].endswith(
        "incompatible types FLOAT16 and FLOAT (shapes [2 2] and [2 2])"
    )
