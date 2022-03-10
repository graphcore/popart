# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional
import popart._internal.ir as _ir
from popxl.context import get_current_context
from popxl.tensor import Tensor
from .collectives import CommGroup
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set


def replicated_all_gather(t: Tensor,
                          group: Optional[CommGroup] = None,
                          link: Optional[Tensor] = None) -> Tensor:
    """Gather a tensor across replicas such that the output tensor contains the values of the tensor from each replica.

    Args:
        t (Tensor): Tensor to be gathered. Must be rank=1.
        group (Optional[CommGroup]): Replicas to gather from. Defaults to All replicas.
        link (Optional[Tensor]): The tensor to link the collective operation with other collective
            operations for replicated tensor sharding. All collective operations, whose link tensor
            leads to the same root tensor in the graph, are added to the same replicated tensor
            sharding group such that their input/output tensors have compatible data orders.
            (such that elementwise operations on the sharded tensor give the semantically correct result).
            The input is optional, if omitted, the group will be determined heuristically.
            The tensor is used for graph traversal only and not consumed by the Op, therefore
            shape and data type do not matter.
    Returns:
        Tensor: Gathered tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if group is None:
        group = CommGroup()

    ins: Dict[int, str] = {0: t.id}

    if link is not None:
        check_in_graph(g, link=link)
        check_tensor_ipu_and_tile_set(t=t, link=link)
        ins[1] = link.id

    settings = ctx._get_op_settings('replicated_all_gathered')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedAllGather", 1,
                                  _ir.NumInputs(1, 2), 1)

    op = pb_g.createConnectedOp_ReplicatedAllGatherOp(
        ins, {0: g._create_tensor_id(t.name + "_all_gathered")}, opid, group,
        settings)

    out = Tensor._from_pb_tensor(op.outTensor(0))

    if t.meta_shape:
        out = out.reshape_(t.meta_shape)

    return out
