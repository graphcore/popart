# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import warnings
from typing import Dict, Optional
import popart._internal.ir as _ir
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Tensor
from .collectives import CommGroup, to_collective_op, CollectiveOps
from popxl.ops.utils import check_in_graph


def replicated_reduce_scatter(
        t: Tensor,
        op: CollectiveOps = 'add',
        group: Optional[ReplicaGrouping] = None,
        configure_output_for_replicated_tensor_sharding: bool = False
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
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    op = to_collective_op(op)

    check_in_graph(g, t=t)

    if group is None:
        comm_group = CommGroup()
    else:
        comm_group = group._to_comm_group()

    settings = ctx._get_op_settings('replicated_reduce_scatter')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedReduceScatter", 1,
                                  _ir.NumInputs(1, 2), 1)

    _op = pb_g.createConnectedOp_ReplicatedReduceScatterOp(
        {0: t.id}, {0: g._create_tensor_id(t.name + "_reduce_scattered")},
        opid, op, comm_group, configure_output_for_replicated_tensor_sharding,
        settings)

    return Tensor._from_pb_tensor(_op.outTensor(0))


def replica_sharded_slice(t: Tensor,
                          group: Optional[ReplicaGrouping] = None) -> Tensor:
    """Take the replicated tensor sharded slice of a Tensor.

    Args:
        t (Tensor): Tensor to be reduced. Inputs will be flattened.
        group (Optional[CommGroup]): Replicas to shard across. Defaults to All replicas.
    Returns:
        Tensor: A slice of the tensor. Always a 1D tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if group is None:
        comm_group = CommGroup()
    else:
        comm_group = group._to_comm_group()

    settings = ctx._get_op_settings('replicated_reduce_scatter')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedReduceScatter", 1,
                                  _ir.NumInputs(1, 2), 1)

    op = pb_g.createConnectedOp_ReplicatedReduceScatterOp(
        {0: t.id}, {0: g._create_tensor_id(t.name + "_reduce_scattered")},
        opid, _ir.CollectiveOperator.Local, comm_group, True, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
