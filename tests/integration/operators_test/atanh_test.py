# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart


def test_atanh(op_tester):
    # create test data
    d1 = np.array(
        [-0.9, -0.77, -0.5, -0.23, -0.1, 0.0, 0.1, 0.23, 0.5, 0.77, 0.9],
        dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.atanh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.arctanh(d1)
        return [out]

    op_tester.setPatterns(['DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_atanh_inplace(op_tester):
    # create test data
    d1 = np.array(
        [-0.9, -0.77, -0.5, -0.23, -0.1, 0.0, 0.1, 0.23, 0.5, 0.77, 0.9],
        dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.atanh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.arctanh(d1)
        return [out]

    op_tester.setPatterns(['InPlace', 'DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_atanh_grad(op_tester):
    # create test data
    d1 = np.array(
        [-0.9, -0.77, -0.5, -0.23, -0.1, 0.0, 0.1, 0.23, 0.5, 0.77, 0.9],
        dtype=np.float32)

    def derivative_atanh(x):
        return 1 / (1 - np.power(x, 2))

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.atanh([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        out = np.arctanh(d1)
        d__o = derivative_atanh(d1) * ref_data.getOutputTensorGrad(0)
        return [out, d__o, None]

    op_tester.setPatterns([
        'SubtractArg1GradOp', 'DivArg0GradOp', 'DivArg1GradOp', 'LogGradOp',
        'MulArgGradOp', 'DecomposeBinaryConstScalar'
    ],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
