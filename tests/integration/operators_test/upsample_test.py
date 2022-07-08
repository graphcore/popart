# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import torch.nn.functional as F
from packaging import version
import numbers


# if the version of torch is greater or equal to 1.5.0, use
# F.interpolate, otherwise use a matching interpolate
# function. This is required as torch versions below 1.5.0
# don't have the `recompute_scale_factor` parameter for
# `F.interpolate` and not all the buildbots appear to have
# an up to date version of torch.
def interpolate(data, scale_factor):
    if version.parse(torch.__version__) >= version.parse("1.5.0"):
        return F.interpolate(
            data, scale_factor=scale_factor, recompute_scale_factor=False
        )
    else:
        if isinstance(scale_factor, numbers.Number):
            scale_factor = [scale_factor]
        scale_factor = [1.0, 1.0] + scale_factor

        result = data
        out_shape = data.shape
        out_shape = [int(i * s) for i, s in zip(out_shape, scale_factor)]

        def resize_nearest(x, dim, size, scale):
            slices = torch.split(x, 1, dim)
            to_concat = []

            to_concat = [slices[int(i / scale)] for i in range(size)]

            return torch.cat(to_concat, dim)

        for i in range(len(out_shape)):
            if data.shape[i] != out_shape[i]:
                result = resize_nearest(result, i, out_shape[i], scale_factor[i])

        return result


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
            s = [i for i in scales[2:]]
            b = interpolate(a, s)
            b.retain_grad()
            o = b * torch.tensor(x_data)

            d__o = ref_data.getOutputTensorGrad(0)
            o.backward(torch.tensor(d__o))
            return [o, a.grad, None]

        op_tester.setPatterns(
            ["MulArgGradOp", "UpsampleToResize"], enableRuntimeAsserts=False
        )
        op_tester.run(
            init_builder, reference, "train", opsets={"ai.onnx": 9, "ai.graphcore": 1}
        )

    run_test([2, 2], [2.0, 3.0])
    run_test([2, 2], [2.5, 2.5])
    run_test([3, 2], [2.5, 2.5])
    run_test([5, 3], [2.3, 2.5])
