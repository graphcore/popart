# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def min(t: Tensor,
        axis: Optional[Union[int, Iterable[int]]] = None,
        keepdims: bool = False) -> Tensor:
    """
    Compute the minimum of the elements of a tensor along axes.

    See also `PyTorch Tensor.min <https://pytorch.org/docs/stable/generated/torch.Tensor.min.html>`__, `ONNX Min <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min>`__.

    Args:
        t (Tensor):
            Tensor to compute minimum of.
        axis (int or list):
            Axis or axes to compute minimum along. If none is provided, all axes will be reduced. If the axis is negative, it counts from the
            last to the first axis.
        keepdims (bool):
            Keep the axis that is being reduced (`True` or not (`False`).

    Returns:
        Tensor
            The reduced tensor containing the minimum of the elements along the axes.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings('ReduceMin')
    opid = _ir.OperatorIdentifier("ai.onnx", "ReduceMin", 1, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_ReduceMinOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("min_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
