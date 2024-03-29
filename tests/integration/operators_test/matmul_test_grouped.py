# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np


def test_matmul_grouped_1(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 1, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_2(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 4, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_3(op_tester):
    d1 = np.random.rand(4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 1, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_4(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(4, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_5(op_tester):
    d1 = np.random.rand(3, 3, 3).astype(np.float32)
    d2 = np.random.rand(3, 3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)
