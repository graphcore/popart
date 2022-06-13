# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Tensor
from .collectives import CommGroup
from popxl.ops.utils import check_in_graph


def replicated_all_gather(t: Tensor,
                          group: Optional[ReplicaGrouping] = None) -> Tensor:
    """Gather a tensor across replicas such that the output tensor contains the values of the tensor from each replica.

    Args:
        t (Tensor): Tensor to be gathered. Must be rank=1.
        group (Optional[ReplicaGrouping]): Replicas to gather from. Defaults to All replicas.
    Returns:
        Tensor: Gathered tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if group is None:
        comm_group = CommGroup()
    else:
        comm_group = group._to_comm_group()

    settings = ctx._get_op_settings('replicated_all_gathered')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedAllGather", 1,
                                  _ir.NumInputs(1, 2), 1)

    _op = pb_g.createConnectedOp_ReplicatedAllGatherOp(
        {0: t.id}, {0: g._create_tensor_id(t.name + "_all_gathered")}, opid,
        comm_group, settings)

    out = Tensor._from_pb_tensor(_op.outTensor(0))

    if t.meta_shape:
        out = out.reshape_(t.meta_shape)

    return out
