# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def dynamic_slice(t: Tensor, index: Tensor, axes: List[int], sizes: List[int],
                  no_overlap: bool) -> Tensor:
    """
    Return a cloned slice of the input tensor.

    "Dynamic" means that the index can be specified
    during runtime.

    A slice along an axis can be defined by the tuple
    ( start, stop, step )
    start - will be equal the index for the respective axis
    stop - will be equal index + size for the respective axis
    step - will equal 1

    Limitations:
    Assuming we would like to slice A with dimension (4, 3)
    - Step other than 1 is not supported (which means t[::2,:] is not supported)
    - Negative slicing is not supported (which means t[:-1,:] is not supported)
    - stop greater than the size of the axis is not supported
     (which means t[:5,:] is not supported)

    Args:
        t: Tensor
            Input tensor.
        index: Tensor
            The indices to start the slice from.
        axes: List[int]
            The axes to slice from.
        sizes: List[int]
            The sizes of the slices for the specified axes.
            For example:
            If index = [1, 2], axes = [0, 3] and sizes = [2, 4], the Tensor will be sliced
            t[1:2, :, :, 2:4]
        no_overlap : bool
            If set to true, then correct gradient backpropagation is only guaranteed if
            each region in the output tensor has exactly one populator
            (operation that writes data to this region).
            There are no run-time or compile-time checks possible to ensure this.
    Returns:
        out: Tensor
            A clone (which means, not a view) of the sliced input tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, index=index)
    check_tensor_ipu_and_tile_set(t=t, index=index)

    settings = ctx._get_op_settings('dynamicslice')
    opid = _ir.OperatorIdentifier("ai.graphcore", "DynamicSlice", 1,
                                  _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_DynamicSliceOp(
        {
            0: t.id,
            1: index.id
        }, {0: g._create_tensor_id(f"dynamic_slice_out")}, opid, axes, sizes,
        no_overlap, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
