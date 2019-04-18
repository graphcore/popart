import numpy as np
import poponnx
import torch
import pytest
import torch.nn.functional as F
from operators_test.op_tester import op_tester


def test_pad_sum1(op_tester):
    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    d2 = np.random.rand(2, 2, 2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i1 = builder.aiOnnx.pad([i1], [0, 2, 0, 0, 0, 0], 'constant', 0)
        i2 = builder.aiOnnx.pad([i2], [0, 0, 0, 0, 2, 0], 'constant', 0)
        o = builder.aiOnnx.sum([i1, i2])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        i1 = np.pad(d1, [(0, 0), (2, 0), (0, 0)], 'constant')
        i2 = np.pad(d2, [(0, 0), (0, 2), (0, 0)], 'constant')
        return [i1 + i2]

    op_tester.passes = ['PadSum']
    op_tester.run(init_builder, reference, 'infer')


def test_pad_sum2(op_tester):
    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    d2 = np.random.rand(2, 2, 2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i1 = builder.aiOnnx.pad([i1], [2, 0, 0, 0, 0, 0], 'constant', 0)
        i2 = builder.aiOnnx.pad([i2], [0, 0, 0, 2, 0, 0], 'constant', 0)
        o = builder.aiOnnx.sum([i1, i2])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        i1 = np.pad(d1, [(2, 0), (0, 0), (0, 0)], 'constant')
        i2 = np.pad(d2, [(0, 2), (0, 0), (0, 0)], 'constant')
        return [i1 + i2]

    op_tester.passes = ['PadSum']
    op_tester.run(init_builder, reference, 'infer')


def test_pad_sum3(op_tester):
    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    d2 = np.random.rand(2, 2, 2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i1 = builder.aiOnnx.pad([i1], [0, 0, 2, 0, 0, 0], 'constant', 0)
        i2 = builder.aiOnnx.pad([i2], [0, 0, 0, 0, 0, 2], 'constant', 0)
        o = builder.aiOnnx.sum([i1, i2])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        i1 = np.pad(d1, [(0, 0), (0, 0), (2, 0)], 'constant')
        i2 = np.pad(d2, [(0, 0), (0, 0), (0, 2)], 'constant')
        return [i1 + i2]

    op_tester.passes = ['PadSum']
    op_tester.run(init_builder, reference, 'infer')


def test_pad_sum4(op_tester):
    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    d2 = np.random.rand(2, 2, 2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i1 = builder.aiOnnx.pad([i1], [0, 0, 3, 0, 0, 0], 'constant', 0)
        i2 = builder.aiOnnx.pad([i2], [0, 0, 0, 0, 0, 3], 'constant', 0)
        o = builder.aiOnnx.sum([i1, i2])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        i1 = np.pad(d1, [(0, 0), (0, 0), (3, 0)], 'constant')
        i2 = np.pad(d2, [(0, 0), (0, 0), (0, 3)], 'constant')
        return [i1 + i2]

    op_tester.passes = ['PadSum']
    op_tester.run(init_builder, reference, 'infer')


def test_pad_sum5(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)
    d3 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i1 = builder.aiOnnx.pad([i1], [2, 6], 'constant', 0)
        i2 = builder.aiOnnx.pad([i2], [4, 4], 'constant', 0)
        i3 = builder.aiOnnx.pad([i3], [6, 2], 'constant', 0)
        o = builder.aiOnnx.sum([i1, i2, i3])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        i1 = np.pad(d1, [(2, 6)], 'constant')
        i2 = np.pad(d2, [(4, 4)], 'constant')
        i3 = np.pad(d3, [(6, 2)], 'constant')
        return [i1 + i2 + i3]

    op_tester.passes = ['PadSum']
    op_tester.run(init_builder, reference, 'infer')


def test_pad_sum6(op_tester):
    """
    The output tensor with be (2, 20, 2)
    This dimensions along the axis=1 that the inputs go are,
    d1 -> [17,18)
    d2 -> [8, 10)
    d3 -> [10, 13)
    d4 -> [3, 7)

    Looks like this:
    ...[..].[][.]....|..
    """
    d1 = np.random.rand(2, 1, 2).astype(np.float32)
    d2 = np.random.rand(2, 2, 2).astype(np.float32)
    d3 = np.random.rand(2, 3, 2).astype(np.float32)
    d4 = np.random.rand(2, 4, 2).astype(np.float32)

    def init_builder(builder):
        def getPadded(start, width, name):
            return builder.aiOnnx.pad(
                [name], [0, start, 0, 0, 20 - start - width, 0], 'constant', 0)

        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i1 = getPadded(17, 1, i1)
        i2 = getPadded(8, 2, i2)
        i3 = getPadded(10, 3, i3)
        i4 = getPadded(3, 4, i4)
        o = builder.aiOnnx.sum([i1, i2, i3, i4])
        builder.addOutputTensor(o)

        return [o]

    def reference(ref_data):
        def getPadded(name, start, width):
            return np.pad(name, [(0, 0), (start, 20 - start - width), (0, 0)],
                          'constant')

        i1 = getPadded(d1, 17, 1)
        i2 = getPadded(d2, 8, 2)
        i3 = getPadded(d3, 10, 3)
        i4 = getPadded(d4, 3, 4)

        return [i1 + i2 + i3 + i4]

    op_tester.passes = ['PadSum']
    op_tester.run(init_builder, reference, 'infer')


def test_pad_sum7(op_tester):
    d1 = np.random.rand(2, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        s1 = builder.aiOnnx.slice([i1], axes=[0], starts=[0], ends=[1])
        s2 = builder.aiOnnx.slice([i1], axes=[0], starts=[3], ends=[4])
        s3 = builder.aiOnnx.slice([i1], axes=[0], starts=[1], ends=[3])
        c1 = builder.aiOnnx.concat([s1, s2, s3], 0)
        u1 = builder.aiOnnx.unsqueeze([c1], axes=[0])

        o = u1
        builder.addOutputTensor(o)
        return [
            o,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        i1 = torch.tensor(d1, requires_grad=True)
        s1 = i1[0:1]
        s2 = i1[3:4]
        s3 = i1[1:3]
        c1 = torch.cat((s1, s2, s3), 0)
        u1 = torch.unsqueeze(c1, 0)
        o = u1

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, i1.grad, None]

    op_tester.passes = ['PadSum']
    op_tester.run(init_builder, reference, 'train')
