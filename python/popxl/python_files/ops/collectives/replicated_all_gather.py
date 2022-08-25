# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Tensor
from popxl.ops.utils import check_in_graph


def replicated_all_gather(t: Tensor, group: Optional[ReplicaGrouping] = None) -> Tensor:
    """Gather a tensor across replicas such that the output tensor contains the values of the tensor from each replica.
    The output will be `(group.size, t.shape)`
    If the input tensor `t` has a meta shape, the output shape will be `t.meta_shape`

    Args:
        t (Tensor): Tensor to be gathered.
        group (Optional[ReplicaGrouping]): Replicas to gather from. Defaults to All replicas.
    Returns:
        Tensor: Gathered tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    group = g.ir.replica_grouping() if group is None else group
    new_shape = t.meta_shape if t.meta_shape else (group.group_size, *t.shape)

    settings = ctx._get_op_settings("replicated_all_gathered")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "ReplicatedAllGather", 1, _ir.NumInputs(1, 2), 1
    )
    out_info = _ir.TensorInfo(t.dtype._pb_dtype, new_shape)

    _op = pb_g.createConnectedOp_ReplicatedAllGatherOp(
        {0: t.id},
        {0: g._create_tensor_id(t.name + "_all_gathered")},
        opid,
        group._pb_replica_grouping,
        settings,
        out_info,
    )

    out = Tensor._from_pb_tensor(_op.outTensor(0))

    return out
