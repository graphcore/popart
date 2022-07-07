# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def max(t: Tensor,
        axis: Optional[Union[int, Iterable[int]]] = None,
        keepdims: bool = False) -> Tensor:
    """
    Compute the maximum value of the elements in a tensor along specified axes.

    See also `PyTorch Tensor.max <https://pytorch.org/docs/stable/generated/torch.Tensor.max.html>`__, `ONNX Max <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max>`__.

    Args:
        t (Tensor):
            Tensor to compute maximum of.
        axis (int or list):
            Axis or axes to computer maximum along. If none is provided all axes will
            be reduced. If an axis is negative it indexes from the
            last to the first axis.
        keepdims (bool):
            Keep the axis that is being reduced (`True`) or not (`False`).

    Returns:
        Tensor:
            The reduced tensor containing the maximum of elements computed along the specified axes.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings('ReduceMax')
    opid = _ir.OperatorIdentifier("ai.onnx", "ReduceMax", 1, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_ReduceMaxOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("max_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def maximum(*ts: Tensor) -> Tensor:
    """
    Compute the elementwise maximum of N tensors.

    Follows NumPy broadcasting rules. Arguments must have the same dtype.

    Args:
        ts (Tensor):
            Tensors to compute the elementwise maximum of.
    Returns:
        Tensor:
            A tensor with the maximum elements from the input tensors.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, **{f"ts_{i}": t for i, t in enumerate(ts)})
    check_tensor_ipu_and_tile_set(**{f"ts_{i}": t for i, t in enumerate(ts)})

    settings = ctx._get_op_settings('maximum')
    opid = _ir.OperatorIdentifier("ai.onnx", "Max", 6, _ir.NumInputs(1, -1), 1)
    op = pb_g.createConnectedOp_MaxOp(
        {i: t.id
         for i, t in enumerate(ts)},
        {
            0: g._create_tensor_id("maximum_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
