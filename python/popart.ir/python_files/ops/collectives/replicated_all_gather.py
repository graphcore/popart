# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Iterable, Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .collectives import CommGroup
from popart.ir.ops.utils import check_in_graph

__all__ = ["replicated_all_gather"]


def replicated_all_gather(t: Tensor,
                          remote_arg: Optional[Tensor] = None,
                          group: Optional[CommGroup] = None) -> Tensor:
    """Gathers tensor `t` across replicas. Output tensor contains in the values of `t` from each replica.

    Args:
        t (Tensor): Tensor to be reduced. Must be rank=1.
        remote_arg (Optional[Tensor]):The tensor associated with a remote variable, 
            returned from remote_variable/remote_replica_sharded_variable.
        group (Optional[CommGroup]): Replicas to gather from. Defaults to All replicas.

    Returns:
        Tensor: Gathered tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if group is None:
        group = CommGroup()

    settings = ctx._get_op_settings('replicated_all_gathered')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedAllGather", 1,
                                  _ir.NumInputs(1, 2), 1)

    ins: Dict[int, str] = {0: t.id}
    if remote_arg is not None:
        ins[1] = remote_arg.id

    op = pb_g.createConnectedOp_ReplicatedAllGatherOp(
        ins, {0: g._create_tensor_id(t.name + "_all_gathered")}, opid, group,
        settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
