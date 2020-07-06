# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester


def test_abs_training(op_tester):
    d1 = np.random.rand(10).astype(np.float32)
    # random numbers is range -3,3
    d1 = 6 * d1 - 3

    # Make sure we have a 0 case
    d1[3] = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.abs([i1], "test_abs")
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + o,
            popart.reservedGradientPrefix() + i1,
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        out = torch.abs(t1)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, d__o, t1.grad]

    op_tester.setPatterns(['AbsGradOp'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_sign_training(op_tester):
    d1 = np.random.rand(10).astype(np.float32)
    # random numbers is range -3,3
    d1 = 6 * d1 - 3

    # Make sure we have a 0 case
    d1[3] = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sign([i1], "test_sign")
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + o,
            popart.reservedGradientPrefix() + i1,
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        out = torch.sign(t1)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, d__o, t1.grad]

    op_tester.setPatterns(['AbsGradOp'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
