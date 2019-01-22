import numpy as np
import torch
from op_tester import op_tester


def test_squeeze(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.squeeze(d1)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_squeeze_limited(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.squeeze(d1, axis=1)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_squeeze_unsorted_axes(op_tester):
    d1 = np.random.rand(2, 1, 3, 1, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[5, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.squeeze(d1, axis=5)
        o = np.squeeze(o, axis=1)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_squeeze_grad(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        i1 = torch.tensor(d1, requires_grad=True)
        o = torch.squeeze(i1)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, i1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_squeeze_limited_grad(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        i1 = torch.tensor(d1, requires_grad=True)
        o = torch.squeeze(i1, dim=1)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, i1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')
