# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

from scatterreduce_bcast_test_basic import (
    scatterreduce_bcast_enabled,
    input_axis_dim_params,
    input_other_dim_params,
    group_size_params,
    axis_params,
)


@pytest.mark.parametrize("input_axis_dim", input_axis_dim_params)
@pytest.mark.parametrize("input_other_dim", input_other_dim_params)
@pytest.mark.parametrize("group_size", group_size_params)
@pytest.mark.parametrize("axis", axis_params)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduction", ["sum"])
@pytest.mark.parametrize("use_torch_scatter_as_reference", [True, False])
def test_scatterreduce_bcast_enabled(
    op_tester,
    input_axis_dim,
    input_other_dim,
    group_size,
    axis,
    include_self,
    reduction,
    use_torch_scatter_as_reference,
):

    scatterreduce_bcast_enabled(
        op_tester,
        input_axis_dim,
        input_other_dim,
        group_size,
        axis,
        include_self,
        reduction,
        use_torch_scatter_as_reference,
    )
