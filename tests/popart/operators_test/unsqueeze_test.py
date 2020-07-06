# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
from op_tester import op_tester
import popart


def test_unsqueeze(op_tester):
    d1 = np.random.rand(2, 3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.unsqueeze([i1], axes=[1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.expand_dims(d1, 1)
        return [o]

    op_tester.setPatterns(['OpToReshape'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_unsqueeze_grad(op_tester):
    d1 = np.random.rand(2, 3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.unsqueeze([i1], axes=[1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        i1 = torch.tensor(d1, requires_grad=True)
        o = torch.unsqueeze(i1, dim=1)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, i1.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'OpToReshape'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
