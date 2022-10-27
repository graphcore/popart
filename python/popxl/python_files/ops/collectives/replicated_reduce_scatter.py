# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Tensor
from .collectives import to_collective_op, CollectiveOps
from popxl.ops.utils import check_in_graph
from .collectives import _rearrange_input, _rearrange_output


def _replicated_reduce_scatter(
    t: Tensor,
    op: _ir.CollectiveOperator,
    group: Optional[ReplicaGrouping],
    configure_output_for_replicated_tensor_sharding: bool,
    suffix: str,
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
        {0: g._create_tensor_id(t.name + suffix)},
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
        t,
        op,
        group,
        configure_output_for_replicated_tensor_sharding,
        "_reduce_scattered",
    )


def replica_sharded_slice(t: Tensor, group: Optional[ReplicaGrouping] = None) -> Tensor:
    """Take the replicated tensor sharded slice of a Tensor.

    Args:
        t (Tensor): Tensor to be reduced. Inputs will be flattened.
        group (Optional[CommGroup]): Replicas to shard across. Defaults to All replicas.
    Returns:
        Tensor: A slice of the tensor. Always a 1D tensor.
    """
    return _replicated_reduce_scatter(
        t, _ir.CollectiveOperator.Local, group, True, "_sharded_slice"
    )


def replicated_slice(
    t: Tensor, axis: int = 0, group: Optional[ReplicaGrouping] = None
) -> Tensor:
    """
    Each replica takes a equal slice of `t` split along axis `axis`.
    e.g. if `t` has shape `(2,4)`, there are two replicas and `axis==0`: the first replica
    will output `[0:1, ...]` and the second replica `[1:2, ...]`.

    This op is similar to `replica_sharded_slice` but differs in that it maintains
    the output shape and does not configure the output for replicated tensor sharding.

    This op is auto-differentiable and it's corresponding grad op is an replicated_all_gather.

    Args:
        t (Tensor): Tensor to split
        axis (int): Axis to slice along
        group (Optional[ReplicaGrouping]): Replica grouping that determines group of replicas
        that slice `t`
    Returns:
        Tensor: A slice of the tensor.
    Raises:
        ValueError: if the group size does not equally divide the axis size
    """
    ctx = get_current_context()
    g = ctx.graph
    group = g.ir.replica_grouping() if group is None else group

    if t.shape[axis] % group.group_size != 0:
        raise ValueError(
            f"Replicated slice must equally divide axis. "
            f"Axis size: {t.shape[axis]}. Replica group size: {group.group_size}"
        )

    t_rearanged, new_shape = _rearrange_input(t, axis)

    if t.shape[axis] / group.group_size == 1:
        new_shape.pop(0)
    else:
        new_shape[0] //= group.group_size

    out = _replicated_reduce_scatter(
        t_rearanged, _ir.CollectiveOperator.Local, group, False, "_replicated_slice"
    )

    y = _rearrange_output(out, new_shape, axis)

    return y
