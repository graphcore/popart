# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from functools import partial
from itertools import product
from typing import Any, List, Optional

import numpy as np
import torch
import torch_scatter
import pytest

import popart
import popart_core

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


def pytorch_scatter_reduce_supported():
    version_to_int = lambda version: int("".join(version.split(".")[:2]))
    current_version = version_to_int(torch.__version__)
    return current_version >= version_to_int("1.13.0")


def idfn(val):
    if isinstance(val, torch.dtype):
        return f"{val}".split(".")[-1]

    return val


def torch_scatter_reduce_to_torch(reduce: str) -> str:
    if reduce == "mul":
        return "prod"
    elif reduce == "max":
        return "amax"
    elif reduce == "min":
        return "amin"

    return reduce


def scatter_reduce_reference(
    src_group: List[torch.Tensor],
    index_group: List[torch.Tensor],
    dim: int = -1,
    out_group: Optional[List[torch.Tensor]] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    group_size: int = 1,
    use_torch_scatter_as_reference: bool = True,
) -> List[torch.Tensor]:

    if reduce == "none":
        use_torch_scatter_as_reference = False
    elif not pytorch_scatter_reduce_supported() and not use_torch_scatter_as_reference:
        use_torch_scatter_as_reference = True

    assert len(src_group) == group_size
    assert len(index_group) == group_size

    results = []
    include_self = out_group is not None

    def prepare_input():
        if include_self:
            input = out_group[group_id]
        else:
            input_size = list(index.shape)
            input_size[dim] = dim_size
            input = torch.zeros(*input_size).float()
        return input

    for group_id in range(group_size):
        src = src_group[group_id]
        index = index_group[group_id]
        out = None
        if out_group is not None:
            out = out_group[group_id]

        if reduce == "none":
            results.append(
                torch.scatter(prepare_input(), dim=dim, index=index, src=src)
            )
        elif use_torch_scatter_as_reference:
            results.append(
                torch_scatter.scatter(
                    src, index, dim, out, dim_size=dim_size, reduce=reduce
                )
            )
        else:
            reduce = torch_scatter_reduce_to_torch(reduce)

            results.append(
                torch.scatter_reduce(
                    input=prepare_input(),
                    dim=dim,
                    index=index,
                    src=src,
                    reduce=reduce,
                    include_self=include_self,
                )
            )

    if group_size == 1:
        return results

    return [torch.stack(results)]


def scatter_reduce_reference_backward(
    d__o: torch.Tensor,
    src_group: List[torch.Tensor],
    index_group: List[torch.Tensor],
    dim: int = -1,
    out_group: Optional[List[torch.Tensor]] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    group_size: int = 1,
    use_torch_scatter_as_reference: bool = True,
) -> List[torch.Tensor]:

    if reduce == "none":
        use_torch_scatter_as_reference = False
    elif not pytorch_scatter_reduce_supported() and not use_torch_scatter_as_reference:
        use_torch_scatter_as_reference = True

    for src_tensor in src_group:
        src_tensor.requires_grad_()

    with_intial_values = out_group is not None

    if with_intial_values and not use_torch_scatter_as_reference:
        for out_tensor in out_group:
            out_tensor.requires_grad_()

    ref_out = scatter_reduce_reference(
        src_group,
        index_group,
        dim=dim,
        out_group=out_group,
        dim_size=dim_size,
        reduce=reduce,
        group_size=group_size,
        use_torch_scatter_as_reference=use_torch_scatter_as_reference,
    )[0]

    ref_out.backward(d__o)

    def get_gradients(group):
        if group is None:
            return None
        gradients = [tensor.grad for tensor in group]
        if None in gradients:
            return None
        return gradients[0] if group_size == 1 else torch.stack(gradients)

    if with_intial_values:
        return [ref_out, get_gradients(src_group), get_gradients(out_group), d__o]

    return [ref_out, get_gradients(src_group), d__o]


def scatter_reduce_builder(
    builder: Any,
    src_group: List[torch.Tensor],
    index_group: List[torch.Tensor],
    dim: int = -1,
    out_group: Optional[List[torch.Tensor]] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    group_size: int = 1,
    backward: bool = False,
):

    with_intial_values = out_group is not None

    if group_size > 1:
        scatter_func = partial(
            builder.aiGraphcore.groupedscatterreduce, group_size=group_size
        )
        if dim >= 0:
            dim += 1
        src = torch.stack(src_group)
        index = torch.stack(index_group)
        if with_intial_values:
            initial = torch.stack(out_group)
    else:
        scatter_func = builder.aiGraphcore.scatterreduce
        src = src_group[0]
        index = index_group[0]
        if with_intial_values:
            initial = out_group[0]

    D = builder.addInputTensor(src.numpy())
    I = builder.addInputTensor(index.numpy().astype(np.uint32))
    input_tensors = [D, I]

    if with_intial_values:
        V = builder.addInputTensor(initial.numpy())
        input_tensors.append(V)

    out = scatter_func(
        input_tensors, axis=dim, axis_size=dim_size, reduction=reduction_map[reduce]
    )

    builder.addOutputTensor(out)

    out_tensors = [out]
    if backward:
        out_tensors.append(popart.reservedGradientPrefix() + D)
        if with_intial_values:
            out_tensors.append(popart.reservedGradientPrefix() + V)
        out_tensors.append(popart.reservedGradientPrefix() + out)

    return out_tensors


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
    axsz = int(torch.max(index)) + 1
    group_size = 2 if grouped else 1

    init_func = torch.rand if init_values else torch.zeros
    initial_values = init_func(axsz, 10, 64)
    initial_values.transpose_(0, axis)
    initial_values = initial_values.contiguous()

    def broadcast_index():
        sz = 3 * [1]
        sz[axis] = -1
        return index.view(sz).expand_as(src).contiguous()

    expanded_index = broadcast_index()
    if broadcast:
        index = expanded_index

    if grouped:
        grouped_axis = axis
        if axis >= 0:
            grouped_axis = grouped_axis + 1

    src_group = [src] if group_size == 1 else [src, src.clone().detach()]
    initial_group = None
    if init_values:
        initial_group = (
            [initial_values]
            if group_size == 1
            else [initial_values, initial_values.clone().detach()]
        )

    def init_builder(builder):
        index_group = [index] * group_size
        return scatter_reduce_builder(
            builder,
            src_group,
            index_group,
            dim=axis,
            out_group=initial_group,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
            backward=True,
        )

    def reference(ref_data):
        index_group = [expanded_index] * group_size

        return scatter_reduce_reference_backward(
            torch.tensor(ref_data.getOutputTensorGrad(0)),
            src_group,
            index_group,
            dim=axis,
            out_group=initial_group,
            dim_size=axsz,
            reduce=reduction,
            group_size=group_size,
            use_torch_scatter_as_reference=not init_values,
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
