# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Optional
import popart._internal.ir as _ir
from popxl.context import get_current_context
from popxl.tensor import Tensor
from .collectives import CommGroup, to_collective_op, CollectiveOps
from popxl.ops.utils import check_in_graph


def replicated_reduce_scatter(
        t: Tensor,
        op: CollectiveOps = 'add',
        group: Optional[CommGroup] = None,
        configure_output_for_replicated_tensor_sharding: bool = False,
        link: Optional[Tensor] = None) -> Tensor:
    """Reduces tensor `t` across replicas. Each replica will only receive a unique slice of `t`.

    Args:
        t (Tensor): Tensor to be reduced. Inputs will be flattened.
        op (str, optional): Operation to reduce with. Defaults to 'add'.
            Options: 'add', 'mean', 'mul', 'min', 'max', 'and', 'or', 'square_add', 'local'.
        group (Optional[CommGroup]): Replicas to reduce across. Defaults to All replicas.
        configure_output_for_replicated_tensor_sharding (Optional[bool]): Configures the output to be a replica sharded tensor. Defaults to false.
            Replicated tensor sharded tensors do not follow the data element order of the original tensor, and can only be
            used in operations that belong to the same replicated tensor sharding group, where all tensor inputs
            follow the same data order.
        link (Optional[Tensor]): The tensor to link the collective operation with other collective
            operations for replicated tensor sharding. All collective operations, whose link tensor
            leads to the same root tensor in the graph, are added to the same replicated tensor
            sharding group such that their input/output tensors have compatible data orders.
            (such that elementwise operations on the sharded tensor give the semantically correct result).
            The input is optional, if omitted, the group will be determined heuristically.
            The tensor is used for graph traversal only and not consumed by the Op, therefore
            shape and data type do not matter.
    Returns:
        Tensor: A slice of the reduced tensor. Always a 1D tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    op = to_collective_op(op)

    check_in_graph(g, t=t)

    if group is None:
        group = CommGroup()

    settings = ctx._get_op_settings('replicated_reduce_scatter')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReplicatedReduceScatter", 1,
                                  _ir.NumInputs(1, 2), 1)

    ins: Dict[int, str] = {0: t.id}
    if link is not None:
        ins[1] = link.id

    op = pb_g.createConnectedOp_ReplicatedReduceScatterOp(
        ins, {0: g._create_tensor_id(t.name + "_reduce_scattered")}, opid, op,
        group, configure_output_for_replicated_tensor_sharding, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
