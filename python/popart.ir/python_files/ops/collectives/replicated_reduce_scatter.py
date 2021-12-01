# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .collectives import CommGroup, to_collective_op, CollectiveOps
from popart.ir.ops.utils import check_in_graph

__all__ = ["replicated_reduce_scatter"]


def replicated_reduce_scatter(t: Tensor,
                              op: CollectiveOps = 'add',
                              group: Optional[CommGroup] = None) -> Tensor:
    """Reduces tensor `t` across replicas. Each replica will only receive a unique slice of `t`.

    Args:
        t (Tensor): Tensor to be reduced. Inputs will be flattened.
        op (str, optional): Operation to reduce with. Defaults to 'add'.
            Options: 'add', 'mean', 'mul', 'min', 'max', 'and', 'or', 'square_add', 'local'.
        group (Optional[CommGroup], optional): Replicas to reduce across. Defaults to All replicas.

    Returns:
        Tensor: A slice of the reduced tensor. Always a 1D tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    op = to_collective_op(op)

    check_in_graph(g, t)

    if group is None:
        group = CommGroup()

    settings = ctx._get_op_settings('replicated_reduce_scatter')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedReduceScatter", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReplicatedReduceScatterOp(
        {0: t.id}, {0: g._create_tensor_id(t.name + "_reduce_scattered")},
        opid, op, group, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
