# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np


def test_prelu_basic(op_tester):
    data = np.random.uniform(-1, 1, [3, 4, 5]).astype(np.float32)
    slope = np.random.uniform(-1, 1, [3, 4, 5]).astype(np.float32)
    # make sure data and slope have both positive and negative values
    data[0][0][0] = -1.0
    data[0][0][1] = 1.0
    slope[0][0][2] = -0.5
    slope[0][0][3] = 0.5

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.addInputTensor(slope)
        o = builder.aiOnnx.prelu([d, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.clip(data, 0, np.inf) + np.clip(data, -np.inf, 0) * slope
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_prelu_broadcast(op_tester):
    data = np.random.uniform(-1, 1, [3, 4, 5]).astype(np.float32)
    slope = np.random.uniform(-1, 1, [5]).astype(np.float32)
    # make sure data and slope have both positive and negative values
    data[0][0][0] = -1.0
    data[0][0][1] = 1.0
    slope[2] = -0.5
    slope[3] = 0.5

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.addInputTensor(slope)
        o = builder.aiOnnx.prelu([d, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.clip(data, 0, np.inf) + np.clip(data, -np.inf, 0) * slope
        return [o]

    op_tester.run(init_builder, reference, 'infer')
