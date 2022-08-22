# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Tensor
from .collectives import to_collective_op, CollectiveOps
from popxl.ops.utils import check_in_graph


def _replicated_reduce_scatter(
    t: Tensor,
    op: _ir.CollectiveOperator,
    group: Optional[ReplicaGrouping],
    configure_output_for_replicated_tensor_sharding: bool,
) -> Tensor:
    """Construct a replicated reduce scatter op.

    This is an internal-only function targeted by public functions in this file.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    group = g.ir.replica_grouping() if group is None else group

    settings = ctx._get_op_settings("replicated_reduce_scatter")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "ReplicatedReduceScatter", 1, _ir.NumInputs(1, 2), 1
    )

    _op = pb_g.createConnectedOp_ReplicatedReduceScatterOp(
        {0: t.id},
        {0: g._create_tensor_id(t.name + "_reduce_scattered")},
        opid,
        op,
        group._pb_replica_grouping,
        configure_output_for_replicated_tensor_sharding,
        settings,
    )

    return Tensor._from_pb_tensor(_op.outTensor(0))


def replicated_reduce_scatter(
    t: Tensor,
    op: CollectiveOps = "add",
    group: Optional[ReplicaGrouping] = None,
    configure_output_for_replicated_tensor_sharding: bool = False,
) -> Tensor:
    """Reduce a tensor across replicas with each replica receiving a unique slice of the tensor.

    Args:
        t (Tensor): Tensor to be reduced. Inputs will be flattened.
        op (str, optional): Operation to reduce with. Defaults to 'add'.
            Options: 'add', 'mean', 'mul', 'min', 'max', 'and', 'or', 'square_add'.
        group (Optional[CommGroup]): Replicas to reduce across. Defaults to All replicas.
        configure_output_for_replicated_tensor_sharding (Optional[bool]): Configures the output to be a replica sharded tensor. Defaults to false.
            Replicated tensor sharded tensors do not follow the data element order of the original tensor, and can only be
            used in operations that belong to the same replicated tensor sharding group, where all tensor inputs
            follow the same data order.
    Returns:
        Tensor: A slice of the reduced tensor. Always a 1D tensor.
    """
    op = to_collective_op(op)
    return _replicated_reduce_scatter(
        t, op, group, configure_output_for_replicated_tensor_sharding
    )


def replica_sharded_slice(t: Tensor, group: Optional[ReplicaGrouping] = None) -> Tensor:
    """Take the replicated tensor sharded slice of a Tensor.

    Args:
        t (Tensor): Tensor to be reduced. Inputs will be flattened.
        group (Optional[CommGroup]): Replicas to shard across. Defaults to All replicas.
    Returns:
        Tensor: A slice of the tensor. Always a 1D tensor.
    """
    return _replicated_reduce_scatter(t, _ir.CollectiveOperator.Local, group, True)
