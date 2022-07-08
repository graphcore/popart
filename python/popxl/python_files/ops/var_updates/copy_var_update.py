# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor

from ..utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def copy_var_update_(t: Tensor, X: Tensor) -> Tensor:
    """
    Update a tensor in-place by copying the tensor containing the updater values.

    Args:
        t: Tensor
            Tensor to be updated.
        X: Tensor
            Value to update the variable
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, X=X)
    check_tensor_ipu_and_tile_set(t=t, X=X)

    settings = ctx._get_op_settings("copy_var_update")
    op = pb_g.createConnectedOp_CopyVarUpdateOp(
        {0: t.id, 1: X.id},
        {0: g._create_tensor_id("updated__" + t.name)},
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
