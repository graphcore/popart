# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import popart_core
import torch
import pytest
from itertools import product

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

reductions = ["sum", "max", "min"]
dtypes = [torch.float32, torch.float16, torch.int]

reduction_map = {
    "sum": popart_core.ScatterReduction.Sum,
    "max": popart_core.ScatterReduction.Max,
    "min": popart_core.ScatterReduction.Min,
    "none": popart_core.ScatterReduction.NoReduction,
}

################################################################################
# Test cases copied and from the torch_scatter package:
#
# https://github.com/rusty1s/pytorch_scatter/blob/2.0.9/test/test_scatter.py
#
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
torch_scatter_testcases = [
    {
        "src": [1, 3, 2, 4, 5, 6],
        "index": [0, 1, 0, 1, 1, 3],
        "dim": 0,
        "sum": [3, 12, 0, 6],
        "add": [3, 12, 0, 6],
        "mul": [2, 60, 1, 6],
        "mean": [1.5, 4, 0, 6],
        "min": [1, 3, 0, 6],
        "arg_min": [0, 1, 6, 5],
        "max": [2, 5, 0, 6],
        "arg_max": [2, 4, 6, 5],
    },
    {
        "src": [[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]],
        "index": [0, 1, 0, 1, 1, 3],
        "dim": 0,
        "sum": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "add": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "mul": [[1 * 3, 2 * 4], [5 * 7 * 9, 6 * 8 * 10], [1, 1], [11, 12]],
        "mean": [[2, 3], [7, 8], [0, 0], [11, 12]],
        "min": [[1, 2], [5, 6], [0, 0], [11, 12]],
        "arg_min": [[0, 0], [1, 1], [6, 6], [5, 5]],
        "max": [[3, 4], [9, 10], [0, 0], [11, 12]],
        "arg_max": [[2, 2], [4, 4], [6, 6], [5, 5]],
    },
    {
        "src": [[1, 5, 3, 7, 9, 11], [2, 4, 8, 6, 10, 12]],
        "index": [[0, 1, 0, 1, 1, 3], [0, 0, 1, 0, 1, 2]],
        "dim": 1,
        "sum": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "add": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "mul": [[1 * 3, 5 * 7 * 9, 1, 11], [2 * 4 * 6, 8 * 10, 12, 1]],
        "mean": [[2, 7, 0, 11], [4, 9, 12, 0]],
        "min": [[1, 5, 0, 11], [2, 8, 12, 0]],
        "arg_min": [[0, 1, 6, 5], [0, 2, 5, 6]],
        "max": [[3, 9, 0, 11], [6, 10, 12, 0]],
        "arg_max": [[2, 4, 6, 5], [3, 4, 5, 6]],
    },
    {
        "src": [[[1, 2], [5, 6], [3, 4]], [[10, 11], [7, 9], [12, 13]]],
        "index": [[0, 1, 0], [2, 0, 2]],
        "dim": 1,
        "sum": [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        "add": [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        "mul": [[[3, 8], [5, 6], [1, 1]], [[7, 9], [1, 1], [120, 11 * 13]]],
        "mean": [[[2, 3], [5, 6], [0, 0]], [[7, 9], [0, 0], [11, 12]]],
        "min": [[[1, 2], [5, 6], [0, 0]], [[7, 9], [0, 0], [10, 11]]],
        "arg_min": [[[0, 0], [1, 1], [3, 3]], [[1, 1], [3, 3], [0, 0]]],
        "max": [[[3, 4], [5, 6], [0, 0]], [[7, 9], [0, 0], [12, 13]]],
        "arg_max": [[[2, 2], [1, 1], [3, 3]], [[1, 1], [3, 3], [2, 2]]],
    },
    {
        "src": [[1, 3], [2, 4]],
        "index": [[0, 0], [0, 0]],
        "dim": 1,
        "sum": [[4], [6]],
        "add": [[4], [6]],
        "mul": [[3], [8]],
        "mean": [[2], [3]],
        "min": [[1], [2]],
        "arg_min": [[0], [0]],
        "max": [[3], [4]],
        "arg_max": [[1], [1]],
    },
    {
        "src": [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
        "index": [[0, 0], [0, 0]],
        "dim": 1,
        "sum": [[[4, 4]], [[6, 6]]],
        "add": [[[4, 4]], [[6, 6]]],
        "mul": [[[3, 3]], [[8, 8]]],
        "mean": [[[2, 2]], [[3, 3]]],
        "min": [[[1, 1]], [[2, 2]]],
        "arg_min": [[[0, 0]], [[0, 0]]],
        "max": [[[3, 3]], [[4, 4]]],
        "arg_max": [[[1, 1]], [[1, 1]]],
    },
]
# End copied test cases from torch-scatter
################################################################################


def idfn(val):
    if isinstance(val, torch.dtype):
        return f"{val}".split(".")[-1]

    return val


@tu.requires_ipu_model
@pytest.mark.parametrize(
    "test,reduction,dtype",
    product(torch_scatter_testcases, reductions, dtypes),
    ids=idfn,
)
def test_scatterreduce_basic(op_tester, test, reduction, dtype):
    src = torch.tensor(test["src"], dtype=dtype)
    index = torch.tensor(test["index"]).long()
    axis = test["dim"]
    axsz = torch.max(index).item() + 1

    if index.dim() > 1:
        for _ in range(index.dim(), src.dim()):
            index = index.unsqueeze(-1)

        index = index.expand_as(src)

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce(
            [D, I], axis=axis, axis_size=axsz, reduction=reduction_map[reduction]
        )
        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        expected = torch.tensor(test[reduction], dtype=dtype)
        return [expected]

    op_tester.run(init_builder, reference)


def test_scatterreduce_index_broadcasted(op_tester):
    src = torch.tensor([[2, 4, 9], [5, 3, 1], [1, 8, 6], [0, 2, 7]]).float()
    index = torch.tensor([[2, 1, 0], [1, 0, 1], [0, 2, 1], [1, 2, 2]]).long()
    axsz = torch.max(index).item() + 1

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce([D, I], axis_size=axsz, axis=0)
        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        ref = torch.zeros(axsz, src.shape[1])
        ref.scatter_add_(dim=0, index=index, src=src)
        return [ref]

    op_tester.run(init_builder, reference)


@pytest.mark.parametrize("reduction", ["max", "min"])
def test_scatterreduce_repro(op_tester, reduction):
    src = torch.linspace(-1, 1, 16).view(-1, 2).T.contiguous()
    print(src)
    index = torch.zeros_like(src).long()
    axsz = torch.max(index).item() + 1

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce(
            [D, I], axis_size=axsz, axis=0, reduction=reduction_map[reduction]
        )
        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        reducer = torch.amin if reduction == "min" else torch.amax
        ref = reducer(src, dim=0, keepdim=True)
        return [ref]

    op_tester.run(init_builder, reference)


@pytest.mark.parametrize("reduction", reductions)
def test_scatterreduce_training(op_tester, reduction):
    src = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3]).float()
    index = torch.tensor([0, 0, 1, 0, 2, 2, 3, 3]).long()
    axsz = torch.max(index).item() + 1

    def torch_scatter_reduce(src, index, out, reduction):
        # Note this can be removed once we can move to torch 1.13 or later.
        # As of June 22 2022 the pytorch scatter_reduce method is in beta.
        if reduction == "sum":
            return out.scatter_add(dim=0, index=index, src=src)

        reducer = torch.amin if reduction == "min" else torch.amax

        for idx in index.unique():
            out[idx] = reducer(src[index == idx])

        return out

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce(
            [D, I], axis_size=axsz, reduction=reduction_map[reduction]
        )
        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + D,
            popart.reservedGradientPrefix() + out,
        ]

    def reference(ref_data):
        src.requires_grad_()
        ref = torch.zeros(axsz)
        ref = torch_scatter_reduce(src, index, ref, reduction)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        ref.backward(d__o)
        return [ref, src.grad, d__o]

    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize(
    "axis,broadcast,init_values,reduction",
    product(range(-3, 3), [True, False], [True, False], reductions),
)
def test_scatterreduce_axis(op_tester, axis, broadcast, init_values, reduction):
    torch.manual_seed(0)
    src = torch.rand(6, 10, 64)
    src.transpose_(0, axis)
    src = src.contiguous()

    index = torch.tensor([0, 1, 0, 1, 2, 1]).long()
    index_vec = index
    axsz = int(torch.max(index)) + 1

    init_func = torch.rand if init_values else torch.zeros
    initial_values = init_func(axsz, 10, 64)
    initial_values.transpose_(0, axis)
    initial_values = initial_values.contiguous()

    if broadcast:
        sz = 3 * [1]
        sz[axis] = -1
        index = index.view(sz).expand_as(src).contiguous()

    def torch_reference(src, initials):
        # Note this can be removed once we can move to torch 1.13 or later.
        # As of June 22 2022 the pytorch scatter_reduce method is in beta.
        index = index_vec
        if reduction == "sum":
            return initials.index_add(dim=axis, index=index, source=src)

        aminmax = torch.amin if reduction == "min" else torch.amax
        reducer = lambda x: aminmax(x, dim=0, keepdim=True)

        src = torch.transpose(src, 0, axis)
        out = torch.transpose(initials.clone(), 0, axis)

        if init_values:
            index = torch.cat((torch.arange(0, axsz), index))
            src = torch.vstack((out, src))

        for idx in index.unique():
            out[idx, :, :] = reducer(src[index == idx, :, :])

        return torch.transpose(out, axis, 0)

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))

        if init_values:
            V = builder.addInputTensor(initial_values.numpy())
            out = builder.aiGraphcore.scatterreduce(
                [D, I, V], axis=axis, axis_size=axsz, reduction=reduction_map[reduction]
            )
            builder.addOutputTensor(out)
            return [
                out,
                popart.reservedGradientPrefix() + D,
                popart.reservedGradientPrefix() + V,
                popart.reservedGradientPrefix() + out,
            ]

        out = builder.aiGraphcore.scatterreduce(
            [D, I], axis=axis, axis_size=axsz, reduction=reduction_map[reduction]
        )
        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + D,
            popart.reservedGradientPrefix() + out,
        ]

    def reference(ref_data):
        src.requires_grad_()
        initial_values.requires_grad_()
        ref = torch_reference(src, initial_values)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        ref.backward(d__o)

        if init_values:
            return [ref, src.grad, initial_values.grad, d__o]

        return [ref, src.grad, d__o]

    op_tester.run(init_builder, reference, "train")


def test_scatterreduce_indices_data_different_shape(op_tester):
    # Note how aiGraphcore.scatterreduce differs from the torch implementation,
    # i.e. for the torch op, we need to expand the indices explicitly.
    src = torch.ones((6, 3))
    index = torch.tensor([[0, 1, 2, 3, 4, 0]]).T

    def init_builder(builder):
        data = builder.addInputTensor(src.numpy())
        idx = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce(
            [data, idx],
            axis=0,
            axis_size=5,
        )
        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + data,
            popart.reservedGradientPrefix() + out,
        ]

    def reference(ref_data):
        src.requires_grad = True
        out = torch.zeros((5, 3))
        out = out.scatter_add(src=src, index=index.expand_as(src), dim=0)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        out.backward(d__o)
        return [out, src.grad, d__o]

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
        i = builder.addInputTensor(np.zeros([2, 3, 4], dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([s, i], axis=4, axis_size=5)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(bad_axis, None)

    assert "axis = 4 is outside the acceptable range" in e_info.value.args[0]


def test_scatterreduce_bad_axis_size(op_tester):
    def bad_axis_size(builder):
        s = builder.addInputTensor(np.ones([2, 3, 4], dtype=np.float32))
        i = builder.addInputTensor(np.zeros([2, 3, 4], dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([s, i], axis_size=0)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(bad_axis_size, None)

    assert "axis_size = 0 is not valid" in e_info.value.args[0]


def test_scatterreduce_indices_data_different_bad_shape(op_tester):
    def bad_shape(builder):
        src = builder.addInputTensor(np.ones([6, 3], dtype=np.float32))
        index = builder.addInputTensor(np.zeros([6, 2], dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([src, index], axis=0, axis_size=5)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(bad_shape, None)

    assert "Failed to expand 'indices' shape " in e_info.value.args[0]


def test_scatterreduce_invalid_indices_rank(op_tester):
    def invalid_rank(builder):
        src = builder.addInputTensor(np.ones([3, 4], dtype=np.float32))
        index = builder.addInputTensor(np.zeros([3, 4, 1], dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([src, index], axis=0, axis_size=5)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(invalid_rank, None)

    msg = "Invalid rank for indices input."
    assert msg in e_info.value.args[0]


def test_scatterreduce_partial_broadcasting(op_tester):
    def partial_broadcast(builder):
        src = builder.addInputTensor(np.ones([6, 3, 5], dtype=np.float32))
        index = builder.addInputTensor(np.zeros([1, 3, 5], dtype=np.uint32))
        t = builder.aiGraphcore.scatterreduce([src, index], axis=0, axis_size=5)
        return [t]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(partial_broadcast, None)

    msg = "Partial broadcasting of indices is not currently supported"
    assert msg in e_info.value.args[0]


def test_scatterreduce_none(op_tester):
    num_updates = 16
    num_channels = 32
    axsz = 100
    src = torch.zeros(num_updates, num_channels)
    index = torch.arange(0, num_updates)
    t = torch.ones(axsz, num_channels)

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        T = builder.addInputTensor(t.numpy())
        out = builder.aiGraphcore.scatterreduce(
            [D, I, T], axis_size=axsz, axis=0, reduction=reduction_map["none"]
        )
        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + D,
            popart.reservedGradientPrefix() + T,
            popart.reservedGradientPrefix() + out,
        ]

    def reference(ref_data):
        src.requires_grad_()
        t.requires_grad_()
        t_updated = torch.index_put(t, (index,), src)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        t_updated.backward(d__o)
        return [t_updated, src.grad, t.grad, d__o]

    op_tester.run(init_builder, reference, "train")
