# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import torch


def test_and(op_tester):
    d1 = (np.random.randn(2) > 0).astype(np.bool_)
    d2 = (np.random.randn(2) > 0).astype(np.bool_)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.logical_and([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1, dtype=torch.bool)
        t2 = torch.tensor(d2, dtype=torch.bool)
        out = t1 & t2
        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_broadcast_and(op_tester):
    d1 = (np.random.randn(2, 2) > 0).astype(np.bool_)
    d2 = (np.random.randn(2) > 0).astype(np.bool_)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.logical_and([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1, dtype=torch.bool)
        t2 = torch.tensor(d2, dtype=torch.bool)
        out = t1 & t2
        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_or(op_tester):
    d1 = (np.random.randn(2) > 0).astype(np.bool_)
    d2 = (np.random.randn(2) > 0).astype(np.bool_)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.logical_or([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1, dtype=torch.bool)
        t2 = torch.tensor(d2, dtype=torch.bool)
        out = t1 | t2
        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_broadcast_or(op_tester):
    d1 = (np.random.randn(2, 2) > 0).astype(np.bool_)
    d2 = (np.random.randn(2) > 0).astype(np.bool_)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.logical_or([i1, i2])
        print(o)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1, dtype=torch.bool)
        t2 = torch.tensor(d2, dtype=torch.bool)
        out = t1 | t2
        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_not(op_tester):
    d1 = (np.random.randn(2) > 0).astype(np.bool_)
    print(d1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.logical_not([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return [np.logical_not(d1)]

    op_tester.run(init_builder, reference, step_type="infer")


def test_equal(op_tester):
    d1 = (np.random.randn(2)).astype(np.float32)
    d2 = (np.random.randn(2)).astype(np.float32)
    d2[0] = d1[0]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.equal([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.eq(t1, t2)

        return [out]

    op_tester.run(init_builder, reference, step_type="infer")


def test_broadcast_equal(op_tester):
    d1 = (np.random.randn(2, 2)).astype(np.float32)
    d2 = (np.random.randn(2)).astype(np.float32)

    # d2[0][0] = d1[0]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.equal([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.eq(t1, t2)

        return [out]

    op_tester.run(init_builder, reference, step_type="infer")
