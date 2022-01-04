# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest


def test_scatterreduce_basic(op_tester):
    src = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3]).float()
    index = torch.tensor([0, 0, 1, 0, 2, 2, 3, 3]).long()
    axsz = torch.max(index).item() + 1

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce([D, I], axis_size=axsz)
        builder.addOutputTensor(out)
        return [out]

    def reference(ref_data):
        ref = torch.zeros(axsz)
        ref.scatter_add_(dim=0, index=index, src=src)
        return [ref]

    op_tester.run(init_builder, reference)


def test_scatterreduce_full_indices(op_tester):
    src = torch.tensor([[2, 4, 9], [5, 3, 1], [1, 8, 6], [0, 2, 7]]).float()
    index = torch.tensor([[2, 1, 0], [1, 0, 1], [0, 2, 1], [1, 2, 2]]).long()
    axsz = torch.max(index).item() + 1

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce([D, I], axis_size=axsz, axis=0)
        builder.addOutputTensor(out)
        return [out]

    def reference(ref_data):
        ref = torch.zeros(axsz, src.shape[1])
        ref.scatter_add_(dim=0, index=index, src=src)
        return [ref]

    op_tester.run(init_builder, reference)


def test_scatterreduce_training(op_tester):
    src = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3]).float()
    index = torch.tensor([0, 0, 1, 0, 2, 2, 3, 3]).long()
    axsz = torch.max(index).item() + 1

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce([D, I], axis_size=axsz)
        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + D,
            popart.reservedGradientPrefix() + out
        ]

    def reference(ref_data):
        src.requires_grad_()
        ref = torch.zeros(axsz)
        ref = ref.scatter_add(dim=0, index=index, src=src)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        ref.backward(d__o)
        return [ref, src.grad, d__o]

    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("axis", range(-3, 3))
def test_scatterreduce_axis(op_tester, axis):
    torch.manual_seed(0)
    src = torch.randn(6, 10, 64)
    src.transpose_(0, axis)
    src = src.contiguous()
    index = torch.tensor([0, 1, 0, 1, 2, 1]).long()
    axsz = torch.max(index).item() + 1
    sz = 3 * [1]
    sz[axis] = -1
    index = index.view(sz).expand_as(src).contiguous()

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce([D, I],
                                                axis=axis,
                                                axis_size=axsz)
        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + D,
            popart.reservedGradientPrefix() + out
        ]

    def reference(ref_data):
        src.requires_grad_()
        ref = torch.zeros(axsz, 10, 64)
        ref.transpose_(0, axis)
        ref = ref.scatter_add(dim=axis, index=index, src=src)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        ref.backward(d__o)
        return [ref, src.grad, d__o]

    op_tester.run(init_builder, reference, "train")


def test_scatterreduce_shape_inference():
    builder = popart.Builder()
    s = builder.addInputTensor("FLOAT16", [10, 10, 64])
    i = builder.addInputTensor("UINT32", [10, 10, 64])
    t = builder.aiGraphcore.scatterreduce([s, i], axis=1, axis_size=5)

    assert builder.getTensorShape(t) == [10, 5, 64]
    assert builder.getTensorDtypeString(t) == "float16"


def test_scatterreduce_bad_axis(op_tester):
    def bad_axis(builder):
        s = builder.addInputTensor(np.ones([2, 3, 4], dtype=np.float32))
        i = builder.addInputTensor(np.zeros(10, dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([s, i], axis=4, axis_size=5)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(bad_axis, None)

    assert "axis = 4 is outside the acceptable range" in e_info.value.args[0]


def test_scatterreduce_bad_indices_shape(op_tester):
    def bad_indices_shape(builder):
        s = builder.addInputTensor(np.ones([2, 3, 4], dtype=np.float32))
        i = builder.addInputTensor(np.zeros([1, 2], dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([s, i], axis=0, axis_size=5)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(bad_indices_shape, None)

    assert "Mismatch indices and data shape" in e_info.value.args[0]


def test_scatterreduce_bad_axis_size(op_tester):
    def bad_axis_size(builder):
        s = builder.addInputTensor(np.ones([2, 3, 4], dtype=np.float32))
        i = builder.addInputTensor(np.zeros([2], dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([s, i], axis_size=0)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(bad_axis_size, None)

    assert "axis_size = 0 is not valid" in e_info.value.args[0]
