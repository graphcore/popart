# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)


def test_scaledadd_constant(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiGraphcore.scaledadd([i1, i2], scale0=0.5, scale1=0.8)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=False)
        t2 = torch.tensor(d2, requires_grad=False)
        out = 0.5 * t1 + 0.8 * t2
        return [out]

    op_tester.setPatterns(['PreUniRepl', 'MulArgGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')


def test_scaledadd_tensor(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)
    d3 = np.random.rand(1).astype(np.float32)
    d4 = np.random.rand(1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        o = builder.aiGraphcore.scaledadd([i1, i2, i3, i4])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=False)
        t2 = torch.tensor(d2, requires_grad=False)
        t3 = torch.tensor(d3, requires_grad=False)
        t4 = torch.tensor(d4, requires_grad=False)
        out = t3 * t1 + t4 * t2
        return [out]

    op_tester.setPatterns(['PreUniRepl', 'MulArgGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')
