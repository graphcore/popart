# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from itertools import product

import numpy as np
import pytest
import torch

import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

from scatterreduce_test_basic import (
    scatter_reduce_reference,
    scatter_reduce_reference_backward,
    scatter_reduce_builder,
    reductions,
    dtypes,
    create_test_id,
    reduction_map,
)

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


@tu.requires_ipu_model
@pytest.mark.parametrize(
    "test,reduction,dtype",
    product(torch_scatter_testcases, reductions, dtypes),
    ids=create_test_id,
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
        return scatter_reduce_builder(
            builder, [src], [index], dim=axis, dim_size=axsz, reduce=reduction
        )

    def reference(_):  # ref_data is an unused argument
        expected = torch.tensor(test[reduction], dtype=dtype)
        return [expected]

    op_tester.run(init_builder, reference)


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("reduction", reductions)
def test_scatterreduce_index_broadcasted(op_tester, grouped, reduction):
    src = torch.tensor([[2, 4, 9], [5, 3, 1], [1, 8, 6], [0, 2, 7]]).float()
    index = torch.tensor([[2, 1, 0], [1, 0, 1], [0, 2, 1], [1, 2, 2]]).long()
    axis = 0
    axsz = torch.max(index).item() + 1
    if grouped:
        src_2nd_group = torch.tensor(
            [[2, 4, 9], [5, 3, 1], [1, 8, 6], [0, 2, 7]]
        ).float()
        index_2nd_group = torch.tensor(
            [[0, 1, 2], [1, 0, 1], [1, 2, 0], [2, 2, 1]]
        ).long()
    group_size = 2 if grouped else 1

    src_group = [src, src_2nd_group] if grouped else [src]
    index_group = [index, index_2nd_group] if grouped else [index]

    def init_builder(builder):
        return scatter_reduce_builder(
            builder,
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
        )

    def reference(_):  # ref_data is an unused argument
        return scatter_reduce_reference(
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
        )

    op_tester.run(init_builder, reference)


@pytest.mark.parametrize("reduction", ["max", "min"])
def test_scatterreduce_repro(op_tester, reduction):
    src = torch.linspace(-1, 1, 16).view(-1, 2).T.contiguous()
    index = torch.zeros_like(src).long()
    axsz = torch.max(index).item() + 1
    axis = 0

    def init_builder(builder):
        D = builder.addInputTensor(src.numpy())
        I = builder.addInputTensor(index.numpy().astype(np.uint32))
        out = builder.aiGraphcore.scatterreduce(
            [D, I], axis_size=axsz, axis=axis, reduction=reduction_map[reduction]
        )
        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        return scatter_reduce_reference(
            [src], [index], dim=axis, dim_size=axsz, reduce=reduction
        )

    op_tester.run(init_builder, reference)


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("reduction", reductions)
def test_scatterreduce_training(op_tester, grouped, reduction):
    src = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3]).float()
    index = torch.tensor([0, 0, 1, 0, 2, 2, 3, 3]).long()
    axsz = torch.max(index).item() + 1
    axis = 0
    group_size = 2 if grouped else 1

    if grouped:
        src_2nd_group = torch.tensor([6, 2, 8, 3, 4, 3, 2, 4]).float()
        index_2nd_group = torch.tensor([1, 1, 0, 1, 3, 3, 2, 2]).long()

    index_group = [index] if group_size == 1 else [index, index_2nd_group]
    src_group = [src] if group_size == 1 else [src, src_2nd_group]

    def init_builder(builder):
        return scatter_reduce_builder(
            builder,
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
            backward=True,
        )

    def reference(ref_data):
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))

        return scatter_reduce_reference_backward(
            d__o,
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
        )

    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("reduction", reductions)
@pytest.mark.parametrize("axsz", [2, 5])
def test_scatterreduce_indices_data_different_shape(
    op_tester, grouped, reduction, axsz
):
    if reduction in ["max", "min"]:
        pytest.skip("Wrongly calculated gradient, when reduced values are equal.")

    if axsz == 2:
        pytest.skip("Segfault in min max strategy.")

    # Note how aiGraphcore.scatterreduce differs from the torch implementation,
    # i.e. for the torch op, we need to expand the indices explicitly.
    src = torch.ones((6, 3))
    index = torch.tensor([[0, 1, 1, 3, 2, 0]]).T
    axis = 0
    axsz = axsz
    group_size = 2 if grouped else 1
    src_group = [src] if group_size == 1 else [src, src.clone().detach()]
    index_group = [index.expand_as(src_tensor) for src_tensor in src_group]

    def init_builder(builder):
        return scatter_reduce_builder(
            builder,
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
            backward=True,
        )

    def reference(ref_data):
        return scatter_reduce_reference_backward(
            torch.tensor(ref_data.getOutputTensorGrad(0)),
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
        )

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
        index = builder.addInputTensor(np.zeros([6, 4], dtype=np.uint32))
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
    axis = 0
    axsz = 100
    group_size = 2 if grouped else 1

    src = torch.zeros(num_updates, num_channels)
    index = torch.arange(0, num_updates).view(16, 1).expand(16, 32)
    t = torch.ones(axsz, num_channels)

    src_group = [src] if group_size == 1 else [src, src.clone().detach()]
    initial_group = [t] if group_size == 1 else [t, t.clone().detach()]
    index_group = [index] if group_size == 1 else [index, index.clone().detach()]

    def init_builder(builder):
        return scatter_reduce_builder(
            builder,
            src_group,
            index_group,
            dim=axis,
            out_group=initial_group,
            dim_size=axsz,
            reduce="none",
            group_size=group_size,
            backward=True,
        )

    def reference(ref_data):
        return scatter_reduce_reference_backward(
            torch.tensor(ref_data.getOutputTensorGrad(0)),
            src_group,
            index_group,
            dim=axis,
            out_group=initial_group,
            dim_size=axsz,
            reduce="none",
            group_size=group_size,
        )

    op_tester.run(init_builder, reference, "train")
