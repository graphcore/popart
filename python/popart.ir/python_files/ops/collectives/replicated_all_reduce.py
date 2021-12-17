# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .collectives import CommGroup, to_collective_op, CollectiveOps
from popart.ir.ops.utils import check_in_graph

__all__ = ["replicated_all_reduce", "replicated_all_reduce_"]


def replicated_all_reduce(t: Tensor,
                          op: CollectiveOps = 'add',
                          group: Optional[CommGroup] = None,
                          link: Optional[Tensor] = None) -> Tensor:
    """Reduces tensor `t` across replicas.

    Args:
        t (Tensor): Tensor to be reduced
        op (str, optional): Operation to reduce with. Defaults to 'add'.
            Options: 'add', 'mean', 'mul', 'min', 'max', 'and', 'or', 'square_add', 'local'.
        group (Optional[CommGroup]): Replicas to reduce across. Defaults to All replicas.
        link (Optional[Tensor]): The tensor to link the collective operation with other collective
            operations for replicated tensor sharding. All collective operations, whose link tensor
            leads to the same root tensor in the graph, are added to the same replicated tensor
            sharding group such that their input/output tensors are elementwise compatible data orders
            (such that elementwise operations on the sharded tensor give the semantically correct result).
            The input is optional, if omitted, the group will be determined heuristically.
            The tensor is used for graph traversal only and not consumed by the Op, therefore
            shape and data type do not matter.
            
    Returns:
        Tensor: Reduced tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    op = to_collective_op(op)

    check_in_graph(g, t)

    if group is None:
        group = CommGroup()

    settings = ctx._get_op_settings('replicated_all_reduce')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedAllReduce", 1,
                                  _ir.NumInputs(1, 2), 1)

    ins: Dict[int, str] = {0: t.id}
    if link is not None:
        ins[1] = link.id

    op = pb_g.createConnectedOp_ReplicatedAllReduceOp(
        ins, {0: g._create_tensor_id(t.name + "_all_reduce")}, opid, op, group,
        settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


def replicated_all_reduce_(t: Tensor,
                           op: CollectiveOps = 'add',
                           group: Optional[CommGroup] = None,
                           link: Optional[Tensor] = None) -> Tensor:
    """Reduces tensor `t` across replicas inplace on `t`.

    Args:
        t (Tensor): Tensor to be reduced
        operations for replicated tensor sharding.
        op (str, optional): Operation to reduce with. Defaults to 'add'.
            Options: 'add', 'mean', 'mul', 'min', 'max', 'and', 'or', 'square_add', 'local'.
        group (Optional[CommGroup]): Replicas to reduce across. Defaults to All replicas.
        link (Optional[Tensor]): The tensor to link the collective operation with other collective
            operations for replicated tensor sharding. All collective operations, whose link tensor
            leads to the same root tensor in the graph, are added to the same replicated tensor
            sharding group such that their input/output tensors have compatible data orders.
            (such that elementwise operations on the sharded tensor give the semantically correct result).
            The input is optional, if omitted, the group will be determined heuristically.
            The tensor is used for graph traversal only and not consumed by the Op, therefore
            shape and data type do not matter.
            
    Returns:
        Tensor: Reduced tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    op = to_collective_op(op)

    check_in_graph(g, t)

    if group is None:
        group = CommGroup()

    settings = ctx._get_op_settings('replicated_all_reduce_inplace')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedAllReduceInplace",
                                  1, _ir.NumInputs(1, 2), 1)

    ins: Dict[int, str] = {0: t.id}
    if link is not None:
        ins[1] = remote_arg.id

    op = pb_g.createConnectedOp_ReplicatedAllReduceInplaceOp(
        ins, {0: g._create_tensor_id(t.name + "_all_reduce")}, opid, op, group,
        settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
