# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, convert_optional_float, check_tensor_ipu_and_tile_set


@op_debug_context
def scatter(t: Tensor,
            indices: Tensor,
            values: Tensor,
            axis: int = 0,
            available_memory_proportion: Optional[float] = None) -> Tensor:
    """
    Select multiple elements from an array, given by `indices`, and updates the values from `values`.

    Scatter takes three inputs data, indices, and values of the same rank r >= 1 and an optional
    attribute axis that identifies an axis of data (by default, the outer-most axis, that is axis 0).
    The output of the operation is produced by creating a copy of the input data, and then
    updating its value to values specified by updates at specific index positions specified by indices.
    Its output shape is the same as the shape of data.

    For each entry in values, the target index in data is obtained by combining the corresponding
    entry in indices with the index of the entry itself: the index-value for dimension = axis is
    obtained from the value of the corresponding entry in indices and the index-value for
    dimension != axis is obtained from the index of the entry itself.

    Pseudo example:

    .. code-block:: python

        x1 = x.copy()
        scatter(x1, [1, 2, 3], [-1, -2, -3])
        x2 = x.copy()
        x[1] = -1
        x[2] = -2
        x[3] = -3
        x1 == x2

    Args:
        t: Tensor
            Input tensor
        indices: Tensor
            The indices of the elements to update
        values: Tensor
            The values to update the tensor with
        axis: int
            Which axis to set on. Default is 0.
        available_memory_proportion: Optional[float]
            The maximum proportion of available memory on each tile that this layer
            should consume temporarily during the course of the operation.
            Defaults to 1.0 if not set globally.

    Returns:
        scatter: Tensor
            The tensor with updated values.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, indices=indices, values=values)
    check_tensor_ipu_and_tile_set(t=t, indices=indices, values=values)

    available_memory_proportion = convert_optional_float(
        available_memory_proportion)

    opid = _ir.OperatorIdentifier("ai.onnx", "Scatter", 11, _ir.NumInputs(
        3, 3), 1)
    settings = ctx._get_op_settings("scatter")
    op = pb_g.createConnectedOp_ScatterOp(
        {
            0: t.id,
            1: indices.id,
            2: values.id
        }, {0: g._create_tensor_id("scatter_out")},
        axis_=axis,
        opid=opid,
        available_memory_proportion_=available_memory_proportion,
        settings=settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
