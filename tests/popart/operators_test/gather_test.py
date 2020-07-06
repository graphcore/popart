# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
from op_tester import op_tester
import popart


def test_gather_id_pattern(op_tester):
    d1 = np.array([[-1, -2, -3]]).astype(np.float32)
    d2 = np.array([0]).astype(np.int32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    # T23410: This test doesn't work with inplacing enabled.
    op_tester.inplacing = False
    op_tester.run(init_builder, reference, 'infer')


def test_gather_rank2_1(op_tester):
    d1 = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.array([0, 2]).astype(np.int32)
    d_d1 = np.array([[1.0, 1.0, 1.0], [0, 0, 0], [1.0, 1.0,
                                                  1.0]]).astype(np.float32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_gather_rank2_2(op_tester):
    d1 = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32).reshape(1, 2)
    d_d1 = np.array([[1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, 1.0,
                                                    0]]).astype(np.float32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_gather_rank3_1(op_tester):
    d1 = np.array([[[-1, -2, -3], [4, 5, 6], [7, 8, 9]]]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32)
    d_d1 = np.array([[[1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, 1.0,
                                                     0]]]).astype(np.float32)
    axis = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_gather_rank1_1(op_tester):
    d1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32)
    d_d1 = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_gather_rank1_0(op_tester):
    d1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    d2 = np.array([]).astype(np.int32)
    d_d1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_gather_example1(op_tester):
    d1 = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]).astype(np.float32)
    d2 = np.array([[[0, 1], [1, 2]]]).astype(np.int32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_gather_example2(op_tester):
    d1 = np.array([[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7,
                                                      5.9]]).astype(np.float32)
    d2 = np.array([[0, 2, 0]]).astype(np.int32)
    d_d1 = np.array([[2.0, 0, 1.0], [2.0, 0, 1.0], [2.0, 0,
                                                    1.0]]).astype(np.float32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
