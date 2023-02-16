# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from itertools import product

import pytest
import torch

from scatterreduce_test_basic import (
    scatter_reduce_reference_backward,
    scatter_reduce_builder,
    reductions,
)


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
