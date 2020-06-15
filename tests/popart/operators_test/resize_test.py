# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import torch.nn.functional as F
from op_tester import op_tester


def test_upsample_nearest1(op_tester):
    def run_test(data_shape, scales):
        data = np.random.rand(1, 1, *data_shape).astype(np.float32)

        scales = np.array([1.0, 1.0] + scales, dtype=np.float32)

        def init_builder(builder):
            d = builder.addInputTensor(data)
            s = builder.aiOnnx.constant(scales)
            o = builder.aiOnnx.resize([d, s])
            builder.addOutputTensor(o)
            return [o]

        def reference(ref_data):
            x = torch.tensor(data)
            s = [
                int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])
            ]
            o = F.interpolate(x, s)
            return [o]

        op_tester.run(init_builder, reference, 'infer')

    run_test([2, 2], [2.0, 3.0])
    run_test([2, 2], [2.5, 2.5])
    run_test([3, 2], [2.5, 2.5])
    run_test([5, 3], [2.3, 2.5])


def test_downsample_nearest(op_tester):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
    outShape = [int(dim * scale) for dim, scale in zip(data.shape, scales)]

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.aiOnnx.constant(scales)
        o = builder.aiOnnx.resize([d, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.array([[[[1., 3.]]]], dtype=np.float32)

        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_resize_nearest(op_tester):
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 0.5], dtype=np.float32)
    outShape = [int(dim * scale) for dim, scale in zip(data.shape, scales)]

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.aiOnnx.constant(scales)
        o = builder.aiOnnx.resize([d, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.array([[[
            [1, 3],
            [1, 3],
            [5, 7],
            [5, 7],
        ]]],
                       dtype=np.float32)

        return [out]

    op_tester.run(init_builder, reference, 'infer')
