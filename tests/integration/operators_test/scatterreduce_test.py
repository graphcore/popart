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

reductions = ["sum", "max", "min", "mul"]
dtypes = [torch.float32, torch.float16, torch.int]

reduction_map = {
    "sum": popart_core.ScatterReduction.Sum,
    "max": popart_core.ScatterReduction.Max,
    "min": popart_core.ScatterReduction.Min,
    "mul": popart_core.ScatterReduction.Mul,
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
        "mul": [2, 60, 0, 6],
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
        "mul": [[1 * 3, 2 * 4], [5 * 7 * 9, 6 * 8 * 10], [0, 0], [11, 12]],
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
        "mul": [[1 * 3, 5 * 7 * 9, 0, 11], [2 * 4 * 6, 8 * 10, 12, 0]],
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
        "mul": [[[3, 8], [5, 6], [0, 0]], [[7, 9], [0, 0], [120, 11 * 13]]],
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


@pytest.mark.parametrize("grouped", [True, False])
def test_scatterreduce_index_broadcasted(op_tester, grouped):
    src = torch.tensor([[2, 4, 9], [5, 3, 1], [1, 8, 6], [0, 2, 7]]).float()
    index = torch.tensor([[2, 1, 0], [1, 0, 1], [0, 2, 1], [1, 2, 2]]).long()
    axsz = torch.max(index).item() + 1
    if grouped:
        src_2nd_group = torch.tensor(
            [[2, 4, 9], [5, 3, 1], [1, 8, 6], [0, 2, 7]]
        ).float()
        index_2nd_group = torch.tensor(
            [[0, 1, 2], [1, 0, 1], [1, 2, 0], [2, 2, 1]]
        ).long()

    def init_builder(builder):
        if grouped:
            D = builder.addInputTensor(torch.stack([src, src_2nd_group]).numpy())
            I = builder.addInputTensor(
                torch.stack([index, index_2nd_group]).numpy().astype(np.uint32)
            )
            out = builder.aiGraphcore.groupedscatterreduce(
                [D, I], axis_size=axsz, axis=1, group_size=2
            )
        else:
            D = builder.addInputTensor(src.numpy())
            I = builder.addInputTensor(index.numpy().astype(np.uint32))
            out = builder.aiGraphcore.scatterreduce([D, I], axis_size=axsz, axis=0)
        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        ref = torch.zeros(axsz, src.shape[1])
        ref.scatter_add_(dim=0, index=index, src=src)
        if grouped:
            ref_2nd_group = torch.zeros(axsz, src_2nd_group.shape[1])
            ref_2nd_group.scatter_add_(dim=0, index=index_2nd_group, src=src_2nd_group)
            ref = torch.stack([ref, ref_2nd_group])
        return [ref]

    op_tester.run(init_builder, reference)


@pytest.mark.parametrize("reduction", ["max", "min"])
def test_scatterreduce_repro(op_tester, reduction):
    src = torch.linspace(-1, 1, 16).view(-1, 2).T.contiguous()
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


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("reduction", reductions)
def test_scatterreduce_training(op_tester, grouped, reduction):
    src = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3]).float()
    index = torch.tensor([0, 0, 1, 0, 2, 2, 3, 3]).long()
    axsz = torch.max(index).item() + 1
    if grouped:
        src_2nd_group = torch.tensor([6, 2, 8, 3, 4, 3, 2, 4]).float()
        index_2nd_group = torch.tensor([1, 1, 0, 1, 3, 3, 2, 2]).long()

    def torch_scatter_reduce(out, reduction):
        # Note this can be removed once we can move to torch 1.13 or later.
        # As of June 22 2022 the pytorch scatter_reduce method is in beta.
        if reduction == "sum":
            ref = out.scatter_add(dim=0, index=index, src=src)
            if grouped:
                ref_2nd_group = out.scatter_add(
                    dim=0, index=index_2nd_group, src=src_2nd_group
                )
                ref = torch.stack([ref, ref_2nd_group])
            return ref

        reducer = None
        if reduction == "min":
            reducer = torch.amin
        elif reduction == "max":
            reducer = torch.amax
        elif reduction == "mul":
            reducer = torch.prod

        if grouped:
            out_2nd = out.clone().detach()
        for idx in index.unique():
            out[idx] = reducer(src[index == idx])
            if grouped:
                out_2nd[idx] = reducer(src_2nd_group[index_2nd_group == idx])
        if grouped:
            out = torch.stack([out, out_2nd])
        return out

    def init_builder(builder):
        if grouped:
            D = builder.addInputTensor(torch.stack([src, src_2nd_group]).numpy())
            I = builder.addInputTensor(
                torch.stack([index, index_2nd_group]).numpy().astype(np.uint32)
            )
            out = builder.aiGraphcore.groupedscatterreduce(
                [D, I],
                axis_size=axsz,
                reduction=reduction_map[reduction],
                axis=1,
                group_size=2,
            )
        else:
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
        if grouped:
            src_2nd_group.requires_grad_()
        ref = torch.zeros(axsz)
        ref = torch_scatter_reduce(ref, reduction)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        ref.backward(d__o)
        src_grad = src.grad
        if grouped:
            src_grad = torch.stack([src_grad, src_2nd_group.grad])
        return [ref, src_grad, d__o]

    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize(
    "axis,broadcast,init_values,grouped,reduction",
    product(range(-3, 3), [True, False], [True, False], [True, False], reductions),
)
def test_scatterreduce_axis(
    op_tester, axis, broadcast, init_values, grouped, reduction
):
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
    if grouped:
        grouped_axis = axis
        if axis >= 0:
            grouped_axis = grouped_axis + 1

    def torch_reference(src, initials):
        # Note this can be removed once we can move to torch 1.13 or later.
        # As of June 22 2022 the pytorch scatter_reduce method is in beta.
        index = index_vec
        if reduction == "sum":
            return initials.index_add(dim=axis, index=index, source=src)

        operation = None
        if reduction == "min":
            operation = torch.amin
        elif reduction == "max":
            operation = torch.amax
        elif reduction == "mul":
            operation = torch.prod

        reducer = lambda x: operation(x, dim=0, keepdim=True)

        src = torch.transpose(src, 0, axis)
        out = torch.transpose(initials.clone(), 0, axis)

        if init_values:
            index = torch.cat((torch.arange(0, axsz), index))
            src = torch.vstack((out, src))

        for idx in index.unique():
            out[idx, :, :] = reducer(src[index == idx, :, :])

        return torch.transpose(out, axis, 0)

    def init_builder(builder):
        if grouped:
            D = builder.addInputTensor(torch.stack([src, src]).numpy())
            I = builder.addInputTensor(
                torch.stack([index, index]).numpy().astype(np.uint32)
            )
        else:
            D = builder.addInputTensor(src.numpy())
            I = builder.addInputTensor(index.numpy().astype(np.uint32))

        if init_values:
            if grouped:
                V = builder.addInputTensor(
                    torch.stack([initial_values, initial_values]).numpy()
                )
                out = builder.aiGraphcore.groupedscatterreduce(
                    [D, I, V],
                    axis=grouped_axis,
                    axis_size=axsz,
                    group_size=2,
                    reduction=reduction_map[reduction],
                )
            else:
                V = builder.addInputTensor(initial_values.numpy())
                out = builder.aiGraphcore.scatterreduce(
                    [D, I, V],
                    axis=axis,
                    axis_size=axsz,
                    reduction=reduction_map[reduction],
                )
            builder.addOutputTensor(out)
            return [
                out,
                popart.reservedGradientPrefix() + D,
                popart.reservedGradientPrefix() + V,
                popart.reservedGradientPrefix() + out,
            ]
        if grouped:
            out = builder.aiGraphcore.groupedscatterreduce(
                [D, I],
                axis=grouped_axis,
                axis_size=axsz,
                group_size=2,
                reduction=reduction_map[reduction],
            )
        else:
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
        if grouped:
            src_2nd_group = src.clone().detach()
            src_2nd_group.requires_grad_()
            initial_values_2nd_group = initial_values.clone().detach()
            initial_values_2nd_group.requires_grad_()
            ref_2nd_group = torch_reference(src_2nd_group, initial_values_2nd_group)
            ref = torch.stack([ref, ref_2nd_group])

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        ref.backward(d__o)

        if grouped:
            src_grad = torch.stack([src.grad, src_2nd_group.grad])
            initial_values_grad = torch.stack(
                [initial_values.grad, initial_values_2nd_group.grad]
            )
        else:
            initial_values_grad = initial_values.grad
            src_grad = src.grad

        if init_values:
            return [ref, src_grad, initial_values_grad, d__o]
        return [ref, src_grad, d__o]

    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("grouped", [True, False])
def test_scatterreduce_indices_data_different_shape(op_tester, grouped):
    # Note how aiGraphcore.scatterreduce differs from the torch implementation,
    # i.e. for the torch op, we need to expand the indices explicitly.
    src = torch.ones((6, 3))
    index = torch.tensor([[0, 1, 2, 3, 4, 0]]).T

    def init_builder(builder):
        if grouped:
            data = builder.addInputTensor(torch.stack([src, src]).numpy())
            idx = builder.addInputTensor(
                torch.stack([index, index]).numpy().astype(np.uint32)
            )
        else:
            data = builder.addInputTensor(src.numpy())
            idx = builder.addInputTensor(index.numpy().astype(np.uint32))
        if grouped:
            out = builder.aiGraphcore.groupedscatterreduce(
                [data, idx], axis=1, axis_size=5, group_size=2
            )
        else:
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
        if grouped:
            src_2nd_group = src.clone().detach()
            src_2nd_group.requires_grad = True
        out = torch.zeros((5, 3))
        out = out.scatter_add(src=src, index=index.expand_as(src), dim=0)
        if grouped:
            out_2nd_group = torch.zeros((5, 3))
            out_2nd_group = out_2nd_group.scatter_add(
                src=src_2nd_group, index=index.expand_as(src_2nd_group), dim=0
            )
            out = torch.stack([out, out_2nd_group])
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        out.backward(d__o)
        if grouped:
            src_grad = torch.stack([src.grad, src_2nd_group.grad])
        else:
            src_grad = src.grad
        return [out, src_grad, d__o]

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


@pytest.mark.parametrize("grouped", [True, False])
def test_scatterreduce_none(op_tester, grouped):
    num_updates = 16
    num_channels = 32
    axsz = 100
    src = torch.zeros(num_updates, num_channels)
    index = torch.arange(0, num_updates)
    t = torch.ones(axsz, num_channels)
    if grouped:
        grouped_src = torch.stack([src, src])
        grouped_index = torch.stack([index, index])
        grouped_t = torch.stack([t, t])

    def init_builder(builder):
        if grouped:
            D = builder.addInputTensor(grouped_src.numpy())
            I = builder.addInputTensor(grouped_index.numpy().astype(np.uint32))
            T = builder.addInputTensor(grouped_t.numpy())
            out = builder.aiGraphcore.groupedscatterreduce(
                [D, I, T],
                axis_size=axsz,
                axis=1,
                group_size=2,
                reduction=reduction_map["none"],
            )
        else:
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
        if grouped:
            # Skip first grad to to avoid double accumulation
            t_updated_2d = torch.index_put(t, (index,), src)
        src.requires_grad_()
        t.requires_grad_()
        t_updated = torch.index_put(t, (index,), src)
        if grouped:
            t_updated = torch.stack([t_updated, t_updated_2d])
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        t_updated.backward(d__o)
        if grouped:
            return [
                t_updated,
                torch.stack([src.grad, src.grad]),
                torch.stack([t.grad, t.grad]),
                d__o,
            ]
        return [t_updated, src.grad, t.grad, d__o]

    op_tester.run(init_builder, reference, "train")
