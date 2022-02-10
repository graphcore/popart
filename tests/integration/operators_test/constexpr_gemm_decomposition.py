# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
# gemm = alpha * A * B + beta * c, where onnx gemm node pattern decomposes that into
# binaryConstScalar op which has a tensor and scalar as attribute.
# This can represent, for example, Tensor * scalar, or Tensor / scalar. See binaryconstscalar.hpp.
# This test checks the output of const folding after the DecomposeBinaryConstScalar pattern has been applied.
# gemm_ce_test.cpp checks the const folding actually happens
import numpy as np


def test_gemm_basic(op_tester):
    m = 2
    k = 2
    n = 2
    alpha = 0.6
    beta = 1.3
    transA = 0
    transB = 0

    data_a = np.arange(m * k, dtype=np.float32).reshape((m, k))
    data_b = np.arange(k * n, dtype=np.float32).reshape((k, n))
    data_c = np.arange(m * n, dtype=np.float32).reshape((m, n))

    def init_builder(builder):
        const_a = builder.aiOnnx.constant(data_a)
        const_c = builder.aiOnnx.constant(data_c)
        input_b = builder.addInputTensor(data_b)

        o = builder.aiOnnx.gemm([const_a, input_b, const_c], alpha, beta,
                                transA, transB)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = data_a
        b = data_b
        c = data_c

        if transA:
            a = np.transpose(a)
        if transB:
            b = np.transpose(b)

        o = alpha * np.dot(a, b) + beta * c
        return [o]

    op_tester.setPatterns(['DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')
