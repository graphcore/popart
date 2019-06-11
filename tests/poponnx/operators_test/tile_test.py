import numpy as np
import pytest
import poponnx
import torch
from op_tester import op_tester


def test_tile(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([2, 4, 6]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.tile([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.tile(d1, d2)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_tile_variable_repeats(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([2, 4, 6]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.tile([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.tile(d1, d2)
        return [out]

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')
    assert (e_info.value.args[0].endswith("must be of type Constant"))


def test_tile_invalid_repeat_vals(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([1, 1, -4]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.tile([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.tile(d1, d2)
        return [out]

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')
    assert (e_info.value.args[0].find("has invalid value") != -1)


def test_tile_invalid_repeats_size(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([2, 1, 4, 5]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.tile([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.tile(d1, d2)
        return [out]

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')
    assert (e_info.value.args[0].endswith(
        "should have one element for each dimension of the data tensor"))


def test_tile_grad(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.array([2, 4, 1]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.tile([i1, c])
        builder.addOutputTensor(o)
        return [
            o,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = a.repeat(tuple(d2))
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')
