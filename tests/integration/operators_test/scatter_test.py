# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart


def test_scatter_0(op_tester):
    data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).astype(
        np.float32
    )
    indices = np.array([[1, 0, 2], [0, 2, 1]]).astype(np.int32)
    updates = np.array([[-1.0, -1.1, -1.2], [2.0, 2.1, 2.2]]).astype(np.float32)
    output = np.array([[2.0, -1.1, 0.0], [-1.0, 0.0, 2.2], [0.0, 2.1, -1.2]]).astype(
        np.float32
    )
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        i2 = builder.addInputTensor(indices)
        i3 = builder.addInputTensor(updates)
        o = builder.aiOnnx.scatter([i1, i2, i3], axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i3,
        ]

    def reference(_):  # ref_data is an unused argument
        data_grad = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ).astype(np.float32)
        return [output, data_grad, np.ones_like(updates)]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


def test_scatter_1(op_tester):
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).astype(np.float32)
    indices = np.array([[1, 3]]).astype(np.int32)
    updates = np.array([[-1.1, 2.1]]).astype(np.float32)
    output = np.array([[1.0, -1.1, 3.0, 2.1, 5.0]]).astype(np.float32)
    d_data = np.array([[1.0, 0, 1.0, 0, 1.0]]).astype(np.float32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        i2 = builder.addInputTensor(indices)
        i3 = builder.addInputTensor(updates)
        o = builder.aiOnnx.scatter([i1, i2, i3], axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i3,
        ]

    def reference(_):  # ref_data is an unused argument
        return [output, d_data, np.ones_like(updates)]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


def test_scatter_2(op_tester):
    data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).astype(
        np.float32
    )
    indices = np.array([[1, 0, 2], [0, 2, 1]]).astype(np.int32)
    updates = np.array([[-1.0, -1.1, -1.2], [2.0, 2.1, 2.2]]).astype(np.float32)
    output = np.array([[-1.1, -1, -1.2], [2, 2.2, 2.1], [0.0, 0.0, 0.0]]).astype(
        np.float32
    )
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        i2 = builder.addInputTensor(indices)
        i3 = builder.addInputTensor(updates)
        o = builder.aiOnnx.scatter([i1, i2, i3], axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i3,
        ]

    def reference(_):  # ref_data is an unused argument
        data_grad = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        ).astype(np.float32)
        return [output, data_grad, np.ones_like(updates)]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")
