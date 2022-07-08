# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def logsum(
    t: Tensor, axis: Optional[Union[int, Iterable[int]]] = None, keepdims: bool = False
) -> Tensor:
    """
    Compute the log of summed elements of a tensor along specified axes.

    Supported dtypes: float.

    Args:
        t (Tensor):
            Tensor to compute the log of the sum of elements.
        axis (int or list):
            Axis or axes to compute the log of the sum along. If none is specified all axes will
            be summed. If an axis is negative it indexes from the
            last to the first axis.
        keepdims (bool):
            Keep the axis that is being computed (`True` or not (`False`).

    Returns:
        Tensor:
            A new tensor containing the log of the summed elements along the specified axes.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings("ReduceLogSum")
    opid = _ir.OperatorIdentifier("ai.onnx", "ReduceLogSum", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReduceLogSumOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("logsum_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
