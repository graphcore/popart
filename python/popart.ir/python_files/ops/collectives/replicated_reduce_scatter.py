# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .collectives import CommGroup, to_collective_op, CollectiveOps
from popart.ir.ops.utils import check_in_graph

__all__ = ["replicated_reduce_scatter"]


def replicated_reduce_scatter(t: Tensor,
                              remote_arg: Optional[Tensor] = None,
                              op: CollectiveOps = 'add',
                              group: Optional[CommGroup] = None) -> Tensor:
    """Reduces tensor `t` across replicas. Each replica will only receive a unique slice of `t`.

    Args:
        t (Tensor): Tensor to be reduced. Inputs will be flattened.
        remote_arg (Optional[Tensor]):The tensor associated with a remote variable, 
            returned from remote_variable/remote_replica_sharded_variable.
        op (str, optional): Operation to reduce with. Defaults to 'add'.
            Options: 'add', 'mean', 'mul', 'min', 'max', 'and', 'or', 'square_add', 'local'.
        group (Optional[CommGroup]): Replicas to reduce across. Defaults to All replicas.

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
                                  _ir.NumInputs(1, 2), 1)

    ins: Dict[int, str] = {0: t.id}
    if remote_arg is not None:
        ins[1] = remote_arg.id

    op = pb_g.createConnectedOp_ReplicatedReduceScatterOp(
        ins, {0: g._create_tensor_id(t.name + "_reduce_scattered")}, opid, op,
        group, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
