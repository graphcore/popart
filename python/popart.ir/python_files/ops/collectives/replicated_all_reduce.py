# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .collectives import CommGroup, CollectiveOperator
from popart.ir.ops.utils import check_in_graph

__all__ = ["replicated_all_reduce", "replicated_all_reduce_"]


def replicated_all_reduce(t: Tensor,
                          op: CollectiveOperator = CollectiveOperator.Add,
                          group: Optional[CommGroup] = None) -> Tensor:
    """Reduces tensor `t` across replicas.

    Args:
        t (Tensor): Tensor to be reduced
        op (CollectiveOperator, optional): Operation to reduce with. Defaults to CollectiveOperator.Add.
        group (Optional[CommGroup], optional): Replicas to reduce across. Defaults to All replicas.

    Returns:
        Tensor: Reduced tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if group is None:
        group = CommGroup()

    settings = ctx._get_op_settings('replicated_all_reduce')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedAllReduce", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReplicatedAllReduceOp(
        {0: t.id}, {0: g._create_tensor_id(t.name + "_all_reduce")}, opid, op,
        group, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


def replicated_all_reduce_(t: Tensor,
                           op: CollectiveOperator = CollectiveOperator.Add,
                           group: Optional[CommGroup] = None) -> Tensor:
    """Reduces tensor `t` across replicas inplace on `t`.

    Args:
        t (Tensor): Tensor to be reduced
        op (CollectiveOperator, optional): Operation to reduce with. Defaults to CollectiveOperator.Add.
        group (Optional[CommGroup], optional): Replicas to reduce across. Defaults to All replicas.

    Returns:
        Tensor: Reduced tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if group is None:
        group = CommGroup()

    settings = ctx._get_op_settings('replicated_all_reduce_inplace')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedAllReduceInplace",
                                  1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReplicatedAllReduceInplaceOp(
        {0: t.id}, {0: g._create_tensor_id(t.name + "_all_reduce")}, opid, op,
        group, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
