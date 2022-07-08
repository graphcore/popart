# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import torch


def test_greater(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.greater([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.gt(t1, t2)

        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_broadcast_greater(op_tester):
    d1 = np.random.rand(2, 2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.greater([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.gt(t1, t2)

        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_less(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.less([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.lt(t1, t2)

        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_broadcast_less(op_tester):
    d1 = np.random.rand(2, 2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.less([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.lt(t1, t2)

        return [out]

    op_tester.run(init_builder, reference, step_type="infer")
