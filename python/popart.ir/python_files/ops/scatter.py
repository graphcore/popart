# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
import popart._internal.ir as _ir
from popart.ir.globals import gcg
from popart.ir.tensor import Tensor
from .utils import check_in_graph, convert_optional_float

__all__ = ["scatter"]


def scatter(t: Tensor, indices: Tensor, values: Tensor,
            axis: int = 0) -> Tensor:
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
    ```
    x1 = x.copy()
    scatter(x, [1, 2, 3], [-1, -2, -3])
    x2 = x.copy()
    x[1] = -1
    x[2] = -2
    x[3] = -3
    x1 == x2
    ```

    Args:
        t: Tensor
            Input tensor
        indices: Tensor
            The indices of the elements to update
        values: Tensor
            The values to update the tensor with
        axis: int
            Which axis to set on. Default is 0.
                
    Returns:
        scatter: Tensor
            The tensor with updated values.
    """
    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, t)
    check_in_graph(g, indices)
    check_in_graph(g, values)

    opid = _ir.OperatorIdentifier("ai.onnx", "Scatter", 11, _ir.NumInputs(
        3, 3), 1)
    settings = _ir.Settings(pb_g, "scatter")
    op = pb_g.createConnectedOp_ScatterOp(
        {
            0: t.id,
            1: indices.id,
            2: values.id
        }, {0: g._create_tensor_id("scatter_out")},
        axis_=axis,
        opid=opid,
        settings=settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
