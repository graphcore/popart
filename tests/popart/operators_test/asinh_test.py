# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
from op_tester import op_tester


def test_asinh(op_tester):
    # create test data
    # Notice: as asinh(x) = ln(x + sqrt(x^2 + 1)), absolute precision
    # deteriorates for larger negative numbers as you will have ln(0.0001).
    d1 = np.array([
        -30.0, -20.12, -2.2, -1.5, -0.2, 0.0, 0.234, 1.0, 1.2, 2.0, 3.0, 10.0,
        100.0, 2001.0
    ],
                  dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.asinh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.arcsinh(d1)
        return [out]

    op_tester.setPatterns(['AsinhOpPattern'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_asinh_inplace(op_tester):
    # create test data
    d1 = np.array([
        -30.0, -20.12, -2.2, -1.5, -0.2, 0.0, 0.234, 1.0, 1.2, 2.0, 3.0, 10.0,
        100.0, 2001.0
    ],
                  dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.asinh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.arcsinh(d1)
        return [out]

    op_tester.setPatterns(['InPlace', 'AsinhOpPattern'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_asinh_grad(op_tester):
    # create test data
    d1 = np.array([
        -20.12, -2.2, -1.5, -0.2, 0.0, 0.234, 1.0, 1.2, 2.0, 3.0, 10.0, 100.0,
        2001.0
    ],
                  dtype=np.float32)

    def derivative_asinh(x):
        return 1 / (np.sqrt(np.power(x, 2) + 1))

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.asinh([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        out = np.arcsinh(d1)
        d__o = derivative_asinh(d1)
        return [out, d__o, None]

    op_tester.setPatterns([
        'AsinhOpPattern', 'SubtractArg1GradOp', 'LogGradOp', 'SqrtGradOp',
        'PowArg0GradOp'
    ],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
