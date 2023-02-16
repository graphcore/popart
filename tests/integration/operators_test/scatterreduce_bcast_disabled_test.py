# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from scatterreduce_test_basic import (
    pytorch_scatter_reduce_supported,
    scatter_reduce_reference,
    scatter_reduce_builder,
)

index_axis_dim_params = [4, 8]
index_other_dim_params = [1, 5]
input_axis_dim_params = [8, 10]
input_other_dim_params = [16, 17]
group_size_params = [1, 2]
axis_params = [0, 1]


def scatterreduce_bcast_disabled(
    op_tester,
    index_axis_dim,
    index_other_dim,
    input_axis_dim,
    input_other_dim,
    group_size,
    axis,
    include_self,
    reduction,
):
    if (
        not include_self
        and input_axis_dim in input_axis_dim_params[1:]
        and input_other_dim in input_other_dim_params[1:]
    ):
        pytest.skip("Duplicated test with include_self=False.")

    torch.manual_seed(42)

    index_group = []
    src_group = []
    out_group = []
    axsz = 8
    axis = axis

    for _ in range(group_size):
        src_shape = [16, 7]
        src_shape.insert(axis, 8)

        src_group.append(torch.randn(*src_shape))
        input_shape = [input_other_dim, 7]
        input_shape.insert(axis, input_axis_dim)

        out_group.append(torch.zeros(*input_shape))

        index_shape = [index_other_dim, index_other_dim + 1]
        index_shape.insert(axis, index_axis_dim)
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
            enable_index_broadcast=False,
        )

    def reference(_):
        return scatter_reduce_reference(
            src_group,
            index_group,
            dim=axis,
            dim_size=axsz,
            out_group=out_group,
            reduce=reduction,
            group_size=group_size,
            use_torch_scatter_as_reference=False,
        )

    op_tester.run(init_builder, reference, "infer")


@pytest.mark.skipif(
    not pytorch_scatter_reduce_supported(), reason="requires torch >= 1.13"
)
@pytest.mark.parametrize("index_axis_dim", index_axis_dim_params)
@pytest.mark.parametrize("index_other_dim", index_other_dim_params)
@pytest.mark.parametrize("input_axis_dim", input_axis_dim_params)
@pytest.mark.parametrize("input_other_dim", input_other_dim_params)
@pytest.mark.parametrize("group_size", group_size_params)
@pytest.mark.parametrize("axis", axis_params)
@pytest.mark.parametrize("include_self", [True, False])
def test_scatterreduce_bcast_disabled_sum(
    op_tester,
    index_axis_dim,
    index_other_dim,
    input_axis_dim,
    input_other_dim,
    group_size,
    axis,
    include_self,
):
    scatterreduce_bcast_disabled(
        op_tester,
        index_axis_dim,
        index_other_dim,
        input_axis_dim,
        input_other_dim,
        group_size,
        axis,
        include_self,
        "sum",
    )


@pytest.mark.skipif(
    not pytorch_scatter_reduce_supported(), reason="requires torch >= 1.13"
)
@pytest.mark.parametrize("index_axis_dim", index_axis_dim_params)
@pytest.mark.parametrize("index_other_dim", index_other_dim_params)
@pytest.mark.parametrize("input_axis_dim", input_axis_dim_params)
@pytest.mark.parametrize("group_size", group_size_params)
@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parametrize("reduction", ["mul", "min"])
def test_scatterreduce_bcast_disabled(
    op_tester,
    index_axis_dim,
    index_other_dim,
    input_axis_dim,
    group_size,
    axis,
    reduction,
):
    input_other_dim = 17
    scatterreduce_bcast_disabled(
        op_tester,
        index_axis_dim,
        index_other_dim,
        input_axis_dim,
        input_other_dim,
        group_size,
        axis,
        include_self=True,
        reduction=reduction,
    )
