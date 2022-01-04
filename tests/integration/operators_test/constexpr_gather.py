# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart


def test_gather_id_pattern(op_tester):
    d1 = np.array([[-1, -2, -3]]).astype(np.float32)
    d2 = np.array([0]).astype(np.int32)
    axis = 0

    def init_builder(builder):
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_gather_rank2_1(op_tester):
    d1 = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.array([0, 2]).astype(np.int32)
    d_d1 = np.array([[1.0, 1.0, 1.0], [0, 0, 0], [1.0, 1.0,
                                                  1.0]]).astype(np.float32)
    axis = 0

    def init_builder(builder):
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
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
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
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
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
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
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
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
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.lossReduction = popart.ReductionType.Sum
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
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out, d_d1]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_gather_complex(op_tester):
    axis = 2
    data = np.zeros((5, 6, 3, 2, 1), dtype=np.float32)

    for i in range(np.prod(data.shape)):
        data.reshape(np.prod(data.shape))[i] = i

    indices = np.zeros((2, 3, 1), dtype=np.int32)
    indices[0, 0, 0] = 0
    indices[0, 1, 0] = 2
    indices[0, 2, 0] = 2
    indices[1, 0, 0] = 0
    indices[1, 1, 0] = 2
    indices[1, 2, 0] = 1

    def init_builder(builder):
        constData = builder.aiOnnx.constant(data)
        constIndices = builder.aiOnnx.constant(indices)
        constOut = builder.aiOnnx.gather([constData, constIndices], axis)
        builder.addOutputTensor(constOut)
        return [constOut]

    def reference(ref_data):
        result = np.zeros((5, 6, 2, 3, 1, 2, 1), dtype=np.float32)
        for d0, d1, d2, d3, d4, d5, d6 in itertools.product(
                range(result.shape[0]), range(result.shape[1]),
                range(result.shape[2]), range(result.shape[3]),
                range(result.shape[4]), range(result.shape[5]),
                range(result.shape[6])):

            # Derive the expected value
            value = (d6 + data.shape[4] *
                     (d5 + data.shape[3] *
                      (indices[d2, d3, d4] + data.shape[2] *
                       (d1 + data.shape[1] * d0))))

            result[d0, d1, d2, d3, d4, d5, d6] = value
        return [result]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')
