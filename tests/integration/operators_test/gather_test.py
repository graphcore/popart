# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_id_pattern(op_tester, grouped):
    d1 = np.array([[-1, -2, -3]]).astype(np.float32)
    d2 = np.array([0]).astype(np.int32)
    axis = 0

    if grouped:
        group_size = 2
        d1 = np.stack([d1, d1])
        d2 = np.stack([d2, d2])

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out = np.stack([out_a, out_b])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.setPatterns(["OpToIdentity"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_rank2_1(op_tester, grouped):
    d1 = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.array([0, 2]).astype(np.int32)
    d_d1 = np.array([[1.0, 1.0, 1.0], [0, 0, 0], [1.0, 1.0, 1.0]]).astype(np.float32)
    axis = 0

    if grouped:
        group_size = 2
        d1 = np.stack([d1, d1])
        d2 = np.stack([d2, d2])
        d_d1 = np.stack([d_d1, d_d1])

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out = np.stack([out_a, out_b])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_rank2_2(op_tester, grouped):
    d1 = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32).reshape(1, 2)
    d_d1 = np.array([[1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, 1.0, 0]]).astype(np.float32)
    axis = 1

    if grouped:
        group_size = 2
        d1 = np.stack([d1, d1])
        d2 = np.stack([d2, d2])
        d_d1 = np.stack([d_d1, d_d1])

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out = np.stack([out_a, out_b])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_rank3_1(op_tester, grouped):
    np.random.seed(0)
    d1 = np.random.randn(3, 3, 3).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32)
    axis = 2

    if grouped:
        group_size = 2
        d1 = np.stack([d1, d1])
        d2 = np.stack([d2, d2])
        d_d1 = np.zeros((group_size, 3, 3, 3), dtype=np.float32)
        d_d1[0, :, :, d2] = 1.0
        d_d1[1, :, :, d2] = 1.0
    else:
        d_d1 = np.zeros((3, 3, 3), dtype=np.float32)
        d_d1[:, :, d2] = 1.0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out = np.stack([out_a, out_b])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_rank1_1(op_tester, grouped):
    d1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32)
    d_d1 = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    axis = 0

    if grouped:
        group_size = 3
        d1 = np.stack([d1, d1, d1])
        d2 = np.stack([d2, d2, d2])
        d_d1 = np.stack([d_d1, d_d1, d_d1])

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out_c = np.take(d1[2], d2[2], axis=axis)
            out = np.stack([out_a, out_b, out_c])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_rank1_0(op_tester, grouped):
    d1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    d2 = np.array([]).astype(np.int32)
    d_d1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    axis = 0

    if grouped:
        group_size = 2
        d1 = np.stack([d1, d1])
        d2 = np.stack([d2, d2])
        d_d1 = np.stack([d_d1, d_d1])

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out = np.stack([out_a, out_b])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_example1(op_tester, grouped):
    d1 = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]).astype(np.float32)
    d2 = np.array([[[0, 1], [1, 2]]]).astype(np.int32)
    axis = 0

    if grouped:
        group_size = 2
        d1 = np.stack([d1, d1])
        d2 = np.stack([d2, d2])

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out = np.stack([out_a, out_b])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


@pytest.mark.parametrize("grouped", [True, False])
def test_gather_example2(op_tester, grouped):
    d1 = np.array([[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]]).astype(
        np.float32
    )
    d2 = np.array([[0, 2, 0]]).astype(np.int32)
    d_d1 = np.array([[2.0, 0, 1.0], [2.0, 0, 1.0], [2.0, 0, 1.0]]).astype(np.float32)
    axis = 1

    if grouped:
        group_size = 2
        d1 = np.stack([d1, d1])
        d2 = np.stack([d2, d2])
        d_d1 = np.stack([d_d1, d_d1])

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        if grouped:
            o = builder.aiGraphcore.groupedgather(
                [i1, i2], axis + 1, group_size=group_size
            )
        else:
            o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        if grouped:
            out_a = np.take(d1[0], d2[0], axis=axis)
            out_b = np.take(d1[1], d2[1], axis=axis)
            out = np.stack([out_a, out_b])
        else:
            out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")
