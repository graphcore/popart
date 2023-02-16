# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from scatterreduce_test_basic import (
    scatter_reduce_reference,
    scatter_reduce_builder,
    pytorch_scatter_reduce_supported,
)

index_axis_dim_params = [4, 8]
index_other_dim_params = [1, 5]
input_axis_dim_params = [8, 10]
input_other_dim_params = [16, 17]
group_size_params = [1, 2]
axis_params = [0, 1, 2]


def scatterreduce_bcast_enabled(
    op_tester,
    input_axis_dim,
    input_other_dim,
    group_size,
    axis,
    include_self,
    reduction,
    use_torch_scatter_as_reference,
):
    if (
        not include_self
        and input_axis_dim in input_axis_dim_params[1:]
        and input_other_dim in input_other_dim_params[1:]
    ):
        pytest.skip("Duplicated test with include_self=False.")

    if not use_torch_scatter_as_reference and not pytorch_scatter_reduce_supported():
        pytest.skip("requires torch >= 1.13")

    src_shape = [16, 7]
    src_shape.insert(axis, 8)

    input_shape = [input_other_dim, 7]
    input_shape.insert(axis, input_axis_dim)

    index_shape = [1, 1]
    index_shape.insert(axis, 8)

    if (
        include_self
        and use_torch_scatter_as_reference
        and input_shape != src_shape
        and reduction in ["max", "min", "mul"]
    ):
        pytest.skip("torch_scatter min, max, mul requires input_shape == src_shape")

    torch.manual_seed(42)

    index_group = []
    src_group = []
    out_group = []
    axsz = 8
    axis = axis

    for _ in range(group_size):
        src_group.append(torch.randn(*src_shape))
        if reduction == "mul":
            out_group.append(torch.ones(*input_shape))
        else:
            out_group.append(torch.zeros(*input_shape))
        index_group.append(torch.randint(0, 8, index_shape))

    if not include_self:
        out_group = None

    def init_builder(builder):
        return scatter_reduce_builder(
            builder,
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            out_group=out_group,
            reduce=reduction,
            group_size=group_size,
            enable_index_broadcast=True,
        )

    index_group_reference = [
        index.expand_as(src) for index, src in zip(index_group, src_group)
    ]

    def reference(_):
        return scatter_reduce_reference(
            src_group,
            index_group_reference,
            dim=axis,
            dim_size=axsz,
            out_group=out_group,
            reduce=reduction,
            group_size=group_size,
            use_torch_scatter_as_reference=use_torch_scatter_as_reference,
        )

    op_tester.run(init_builder, reference, "infer")
