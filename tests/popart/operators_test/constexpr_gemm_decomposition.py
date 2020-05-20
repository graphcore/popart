# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
# this test checks the output of const folding after the GemmDecomposition pattern has been applied
# gemm_ce_test.cpp checks the const folding actually happens
import numpy as np
from op_tester import op_tester


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

    def reference(ref_data):
        a = data_a
        b = data_b
        c = data_c

        if transA:
            a = np.transpose(a)
        if transB:
            b = np.transpose(b)

        o = alpha * np.dot(a, b) + beta * c
        return [o]

    op_tester.patterns = ['GemmDecomposition']
    op_tester.run(init_builder, reference, 'infer')
