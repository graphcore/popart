# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Iterable
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def concat(ts: Iterable[Tensor], axis: int = 0) -> Tensor:
    """
    Concatenate tensors along an axis. The result will be copied to a new tensor.

    See also `ONNX Concat <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat>`__.

    Args:
        ts (Iterable[Tensor]):
            Tensors to be concatenated.
        axis (int):
            Axis of `ts` to concatenate along.
    Returns:
        Tensor:
            The concatenated tensors.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    ts = list(ts)

    check_in_graph(g, **{f"ts_{i}": t for i, t in enumerate(ts)})
    check_tensor_ipu_and_tile_set(**{f"ts_{i}": t for i, t in enumerate(ts)})

    settings = ctx._get_op_settings("concat")
    opid = _ir.OperatorIdentifier("ai.onnx", "Concat", 11, _ir.NumInputs(2, -1), 1)
    op = pb_g.createConnectedOp_ConcatOp(
        {i: t.id for i, t in enumerate(ts)},
        {
            0: g._create_tensor_id("concat_out"),
        },
        opid,
        axis,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def concat_(ts: Iterable[Tensor], axis: int = 0) -> Tensor:
    """
    Concatenate tensors along an axis.

    The result will alias both of the input tensors.

    Args:
        ts (Iterable[Tensor]):
            Tensors to be concatenated.
        axis (int):
            Axis of `ts` to concatenate along.
    Returns:
        Tensor:
            The concatenated tensors.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    ts = list(ts)

    check_in_graph(g, **{f"ts_{i}": t for i, t in enumerate(ts)})
    check_tensor_ipu_and_tile_set(**{f"ts_{i}": t for i, t in enumerate(ts)})

    settings = ctx._get_op_settings("concat_inplace")
    op = pb_g.createConnectedOp_ConcatInplaceOp(
        {i: t.id for i, t in enumerate(ts)},
        {
            0: g._create_tensor_id("concat_out"),
        },
        axis,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
