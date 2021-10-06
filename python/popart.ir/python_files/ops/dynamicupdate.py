# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ["dynamicupdate"]


def dynamicupdate(t: Tensor,
                  index: Tensor,
                  t_update: Tensor,
                  axes: List[int],
                  sizes: List[int],
                  noOverlap: bool,
                  updateInInfo: _ir.TensorInfo = _ir.TensorInfo()) -> Tensor:
    """
    Dynamically updates a tensor.

    The word "dynamic" refers to the fact that the index can be specified
    during runtime.

    index, axes and sizes determines the slice of t which will be updated.
    The dimension of this slice and t_update must match.
    A slice along an axis can be defined as by the tuple
    ( start, stop, step )
    start - will be equal the index for the respective axis
    stop - will be equal index + size for the respective axis
    step - will equal 1
   
    Limitations:
    Assuming we would like to update t with dimension (4, 3).
    The slicing of t will have the following limitations:
    - Step other than 1 is not supported (i.e. t[::2,:] is not supported)
    - Negative slicing is not supported (i.e. t[:-1,:] is not supported)
    - stop larger than the size of the axis is not supported
     (i.e. t[:5,:] is not supported)

    Args:
        t: Tensor
            Tensor to update.
        index: Tensor
            The indices to start the slice from.
        t_update: Tensor
            The tensor to update t with.
        axes: List[int]
            The axess of t to make the update at.
        sizes: List[int]
            The sizes of the updates along the specified axes.
            For example:
            If index = [1, 2], axes = [0, 3] and sizes = [2, 4], the Tensor will be updated at
            t[1:2, :, :, 2:4]
        noOverlap : bool
            If set to true, then correct gradient backpropagation is only guaranteed if
            each region in the output tensor has exactly one populator
            (operation that writes data to this region).
            There are no run-time or compile-time checks possible to ensure this.
        updateInInfo : TensorInfo
           The TensorInfo (containing data_type, shape and meta_shape) for the t_update
    Returns:
        out: Tensor
            The updated tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t, index)

    settings = ctx._get_op_settings('dynamicupdate')
    opid = _ir.OperatorIdentifier("ai.graphcore", "DynamicUpdate", 1,
                                  _ir.NumInputs(3, 3), 1)
    op = pb_g.createConnectedOp_DynamicUpdateOp(
        {
            0: t.id,
            1: index.id,
            2: t_update.id
        }, {0: g._create_tensor_id(f"dynamicupdate_out")}, opid, axes, sizes,
        noOverlap, settings, updateInInfo)

    return Tensor._from_pb_tensor(op.outTensor(0))
