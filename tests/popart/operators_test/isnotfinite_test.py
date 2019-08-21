import numpy as np
import torch
from op_tester import op_tester
import popart


def test_isnan(op_tester):
    # create test data
    d1 = np.array([np.inf, -np.inf, 3.14159, np.nan], np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.isnan([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        b = np.isnan(d1)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_isinf(op_tester):
    # create test data
    #d1 = np.random.rand(4).astype(np.float32)
    d1 = np.array([np.inf, -np.inf, 3.14159, np.nan], np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.isinf([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        b = np.isinf(d1)
        return [b]

    op_tester.run(init_builder, reference, 'infer')
