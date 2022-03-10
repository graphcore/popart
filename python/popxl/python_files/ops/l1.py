# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def l1(t: Tensor,
       axis: Optional[Union[int, Iterable[int]]] = None,
       keepdims: bool = False) -> Tensor:
    """
    Compute the sum of the magnitudes of the elements in a tensor (L1 norm) along specified axes.

    Args:
        t: Tensor
            Tensor to compute L1 norm of.
        axis (int or list):
            Axis or axes to compute L1 norm along. If none is provided all elements
            will be normed. If axis is negative it counts from the
            last to the first axis.
        keepdims (bool):
            Keep the axis that is being reduced (`True`) or not (`False`).

    Returns:
        Tensor
            The reduced tensor containing the L1 norm of elements along the specified axes.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings('ReduceL1')
    opid = _ir.OperatorIdentifier("ai.onnx", "ReduceL1", 1, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_ReduceL1Op(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("l1_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
