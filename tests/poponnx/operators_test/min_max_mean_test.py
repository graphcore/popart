import numpy as np
import poponnx
import torch
import pytest

from op_tester import op_tester


def test_mean_training(op_tester):
    d1 = np.random.rand(8, 7).astype(np.float32)
    d2 = np.random.rand(6, 8, 7).astype(np.float32)
    d3 = np.random.rand(6, 1, 7).astype(np.float32)
    d4 = np.random.rand(7).astype(np.float32)
    d5 = np.random.rand(6, 8, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.mean([i1, i2, i3, i4, i5], "test_max")
        builder.addOutputTensor(o)
        return [
            o, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + i4, 'd__' + i5,
            'd__' + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.add(t1, t2)
        out = torch.add(t3, out)
        out = torch.add(t4, out)
        out = torch.add(t5, out)

        out = torch.div(out, 5)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.passes = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train')


# Test with opset 6 - all have to be the same type and shape
def test_mean_training_2(op_tester):
    d1 = np.random.rand(8, 7).astype(np.float32)
    d2 = np.random.rand(8, 7).astype(np.float32)
    d3 = np.random.rand(8, 7).astype(np.float32)
    d4 = np.random.rand(8, 7).astype(np.float32)
    d5 = np.random.rand(8, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.mean([i1, i2, i3, i4, i5], "test_max")
        builder.addOutputTensor(o)
        return [
            o, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + i4, 'd__' + i5,
            'd__' + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.add(t1, t2)
        out = torch.add(t3, out)
        out = torch.add(t4, out)
        out = torch.add(t5, out)

        out = torch.div(out, 5)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.passes = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train', {
        "ai.onnx": 7,
        "ai.graphcore": 1
    })


# Test with opset 6 with incorrect inputs i.e. different shapes
def test_mean_training_3(op_tester):
    d1 = np.random.rand(8, 7).astype(np.float32)
    d2 = np.random.rand(6, 8, 7).astype(np.float32)
    d3 = np.random.rand(6, 1, 7).astype(np.float32)
    d4 = np.random.rand(7).astype(np.float32)
    d5 = np.random.rand(6, 8, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.mean([i1, i2, i3, i4, i5], "test_max")
        builder.addOutputTensor(o)
        return [
            o, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + i4, 'd__' + i5,
            'd__' + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.add(t1, t2)
        out = torch.add(t3, out)
        out = torch.add(t4, out)
        out = torch.add(t5, out)

        out = torch.div(out, 5)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.passes = ['OpToIdentity']

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'train', {
            "ai.onnx": 7,
            "ai.graphcore": 1
        })

    assert (e_info.value.args[0] ==
            "Inputs to ai.onnx.Mean:6 do not all the same type & shape")


def test_max_training(op_tester):
    d1 = np.random.rand(5, 7, 5).astype(np.float32)
    d2 = np.random.rand(7, 5).astype(np.float32)
    d3 = np.random.rand(5).astype(np.float32)
    d4 = np.random.rand(1, 1, 5).astype(np.float32)
    d5 = np.random.rand(5, 1, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.max([i1, i2, i3, i4, i5], "test_max")
        builder.addOutputTensor(o)
        return [
            o, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + i4, 'd__' + i5,
            'd__' + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.max(t1, t2)
        out = torch.max(t3, out)
        out = torch.max(t4, out)
        out = torch.max(t5, out)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.passes = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train')


def test_min_training(op_tester):
    d1 = np.random.rand(2, 3, 4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)
    d3 = np.random.rand(1, 1, 4).astype(np.float32)
    d4 = np.random.rand(2, 1, 4).astype(np.float32)
    d5 = np.random.rand(1, 3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.min([i1, i2, i3, i4, i5], "test_min")
        builder.addOutputTensor(o)
        return [
            o, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + i4, 'd__' + i5,
            'd__' + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.min(t1, t2)
        out = torch.min(t3, out)
        out = torch.min(t4, out)
        out = torch.min(t5, out)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.passes = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train')
