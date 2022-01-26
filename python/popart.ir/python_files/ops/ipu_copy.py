# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['ipu_copy']


@op_debug_context
def ipu_copy(t: Tensor, destination: int,
             source: Optional[int] = None) -> Tensor:
    """
    Copies a Tensor to a virtual graph.

    Args:
        t: Tensor
            Tensor to be copied.
        destination: int
            Ipu for the tensor to be copied to.
        source: Optional[int]
            Ipu for the tensor to be copied from.
            By default, the source will be taken from the producer of `t`.
            If `t` does not have a producer a source MUST be provided.

    Returns:
        t_copied: Tensor
            The copied tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if source is None:
        # Use internal method to infer the input tensor's ipu.
        source = t._pb_tensor.getVirtualGraphIdUnsafe()
        if source == -1:
            raise ValueError(
                f"Could not infer the ipu for Tensor to be copied \"{t}\" . "
                "Please specify `source` when copying for this tensor.")

    settings = ctx._get_op_settings('ipucopy')
    opid = _ir.OperatorIdentifier("ai.graphcore", "IpuCopy", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_IpuCopyOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id(t.name + f"_c{destination}"),
        },
        opid,
        source,
        destination,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
