# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import torch.nn.functional as F
from op_tester import op_tester


# based on test_upsample_nearest_grad from resize_test
def test_upsample_grad(op_tester):
    def run_test(data_shape, scales):
        data = np.random.rand(1, 1, *data_shape).astype(np.float32)

        scales = np.array([1.0, 1.0] + scales, dtype=np.float32)

        x_data_shape = [int(i * j) for i, j in zip(data.shape, scales)]
        x_data = np.random.rand(*x_data_shape).astype(np.float32)

        def init_builder(builder):
            d = builder.addInputTensor(data)
            x = builder.addInputTensor(x_data)
            s = builder.aiOnnx.constant(scales)
            o = builder.aiOnnx.upsample([d, s])
            o = builder.aiOnnx.mul([o, x])
            builder.addOutputTensor(o)
            return [
                o,
                popart.reservedGradientPrefix() + d,
                popart.reservedGradientPrefix() + o,
            ]

        def reference(ref_data):
            a = torch.tensor(data, requires_grad=True)
            s = [
                int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])
            ]
            b = F.interpolate(a, s)
            b.retain_grad()
            o = b * torch.tensor(x_data)

            d__o = ref_data.getOutputTensorGrad(0)
            o.backward(torch.tensor(d__o))
            return [o, a.grad, None]

        op_tester.setPatterns(['MulArgGradOp', 'UpsampleToResize'],
                              enableRuntimeAsserts=False)
        op_tester.run(init_builder,
                      reference,
                      'train',
                      opsets={
                          "ai.onnx": 9,
                          "ai.graphcore": 1
                      })

    run_test([2, 2], [2.0, 3.0])
    run_test([2, 2], [2.5, 2.5])
    run_test([3, 2], [2.5, 2.5])
    run_test([5, 3], [2.3, 2.5])
