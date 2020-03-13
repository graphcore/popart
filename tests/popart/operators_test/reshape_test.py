# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
from op_tester import op_tester


def test_reshape(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([4, 6]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.reshape(d1, d2)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_reshape_neg_one(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([-1, 6]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.reshape(d1, d2)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_reshape_neg_one_error(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([-1, -1, 0]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [o]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None, 'infer')

    assert ('shape input to ReshapeOp can only use -1 to '
            'specify one unknown dimension') in str(e_info.value)


def test_reshape_zeros(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([6, 0]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        s = [i for i in d2]
        for i in range(0, len(s)):
            if s[i] == 0:
                s[i] = d1.shape[i]
        out = np.reshape(d1, s)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_reshape_neg_one_and_zeros(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([-1, 0]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        s = [i for i in d2]
        for i in range(0, len(s)):
            if s[i] == 0:
                s[i] = d1.shape[i]
        out = np.reshape(d1, s)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_reshape_neg_one_and_zeros_grad(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([-1, 0]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        s = [i for i in d2]
        for i in range(0, len(s)):
            if s[i] == 0:
                s[i] = d1.shape[i]

        a = torch.tensor(d1, requires_grad=True)
        o = torch.reshape(a, s)

        d__o = ref_data.getOutputTensorGrad(0)

        o.backward(torch.tensor(d__o))

        return [o, a.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')
