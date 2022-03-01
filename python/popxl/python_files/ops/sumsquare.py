# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def sumsquare(t: Tensor,
              axis: Optional[Union[int, Iterable[int]]] = None,
              keepdims: bool = False) -> Tensor:
    """
    Compute the sum of the squared elements over a given axis.

    Args:
        t (Tensor):
            Tensor to compute sumsquare.
        axis (int or list):
            Axis or axes to sumsquare over. If none is provided all axes will
            be reduced. If axis is negative it counts from the
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

    settings = ctx._get_op_settings('ReduceSumSquare')
    opid = _ir.OperatorIdentifier("ai.onnx", "ReduceSumSquare", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReduceSumSquareOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("sumsquare_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
