# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def l2(t: Tensor,
       axis: Optional[Union[int, Iterable[int]]] = None,
       keepdims: bool = False) -> Tensor:
    """
    Compute L2 norm for elements over a given axis.

    The L2 norm is the square root of the sum of the squares of the elements.

    Args:
        t: Tensor
            Tensor to compute L2 norm.
        axis (int or list):
            Axis or axes to L2 norm over. If none is provided all elements
            will be normed. If axis is negative it counts from the
            last to the the first axis.
        keepdims (bool):
            Keep the axis that is being reduced or not.

    Returns:
        Tensor
            The reduced tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings('ReduceL2')
    opid = _ir.OperatorIdentifier("ai.onnx", "ReduceL2", 1, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_ReduceL2Op(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("l2_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
