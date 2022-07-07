# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def dynamic_update(t: Tensor, index: Tensor, t_update: Tensor,
                   axes: Iterable[int], sizes: Iterable[int],
                   no_overlap: bool) -> Tensor:
    """
    Update a slice of a tensor.

    The name "dynamic" refers to the fact that the index can be specified
    at runtime.

    `index`, `axes` and `sizes` determine the slice of `t` which will be updated.
    The dimensions of this slice and `t_update` must match.
    A slice along an axis can be defined by the tuple
    (`start`, `stop`, `step`) where:

    - `start` is the index for the respective axis
    - `stop` is `index` + `size` for the respective axis
    - `step` equals 1

    Limitations:

    Assuming we would like to update `t` with dimension [4, 3],
    the slicing of `t` will have the following limitations:

    - A `step` other than 1 is not supported (that is, `t[::2,:]` is not supported)
    - Negative slicing is not supported (that is, `t[:-1,:]` is not supported)
    - A value of `stop` larger than the size of the axis is not supported
      (for example, `t[:5,:]` is not supported)

    Args:
        t (Tensor):
            The tensor to update.
        index (Tensor):
            The indices to start the slice from.
        t_update (Tensor):
            The tensor to update `t` with.
        axes (Iterable[int]):
            The axes of `t` to make the update on.
        sizes (Iterable[int]):
            The sizes of the updates along the specified axes.
            For example,
            if `index` = [1, 2], `axes` = [0, 3] and `sizes` = [2, 4], then the tensor will be updated at
            `t[1:2, :, :, 2:4]`.
        no_overlap (bool):
            If set to true, then correct gradient backpropagation is only guaranteed if
            each region in the output tensor has exactly one populator
            (operation that writes data to this region).
            There are no run-time or compile-time checks possible to ensure this.
    Returns:
        Tensor:
            The updated tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, index=index, t_update=t_update)
    check_tensor_ipu_and_tile_set(t=t, index=index, t_update=t_update)

    settings = ctx._get_op_settings('dynamicupdate')
    opid = _ir.OperatorIdentifier("ai.graphcore", "DynamicUpdate", 1,
                                  _ir.NumInputs(3, 3), 1)
    # This ensures that `t` is created by calling `popops::createSliceableTensorFromSlice`
    # with `t_update`.
    # Does the user need control over this?
    settings.inferTensorMappingToFrom = {0: 2}
    op = pb_g.createConnectedOp_DynamicUpdateOp(
        {
            0: t.id,
            1: index.id,
            2: t_update.id
        }, {0: g._create_tensor_id("dynamic_update_out")}, opid, axes, sizes,
        no_overlap, settings, t_update._pb_tensor.info)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def dynamic_update_(t: Tensor, index: Tensor, t_update: Tensor,
                    axes: Iterable[int], sizes: Iterable[int],
                    no_overlap: bool) -> Tensor:
    """
    Update a slice of a tensor in place.

    Dynamically updates tensor `t` in place. The name "dynamic" refers to the
    fact that the index can be specified during runtime.

    `index`, `axes` and `sizes` determine the slice of `t` which will be updated.
    The dimensions of this slice and `t_update` must match.
    A slice along an axis can be defined by the tuple
    (`start`, `stop`, `step`) where:

    - `start` is the index for the respective axis
    - `stop` is `index` + `size` for the respective axis
    - `step` equals 1

    Limitations:

    Assuming we would like to update `t` with dimension [4, 3],
    the slicing of `t` will have the following limitations:

    - A step value other than 1 is not supported (that is, `t[::2,:]` is not supported)
    - Negative slicing is not supported (that is, t[:-1,:] is not supported)
    - A `stop` value larger than the size of the axis is not supported
      (for example, t[:5,:] is not supported)

    Args:
        t (Tensor):
            Tensor to update.
        index (Tensor):
            The indices to start the slice from.
        t_update (Tensor):
            The tensor to update `t` with.
        axes (List[int]):
            The axes of `t` to make the update on.
        sizes (List[int]):
            The sizes of the updates along the specified axes.
            For example,
            if `index` = [1, 2], `axes` = [0, 3] and `sizes` = [2, 4], the tensor will be updated at
            `t[1:2, :, :, 2:4]`.
        no_overlap (bool):
            If set to true, then correct gradient backpropagation is only guaranteed if
            each region in the output tensor has exactly one populator
            (operation that writes data to this region).
            There are no run-time or compile-time checks possible to ensure this.
    Returns:
        Tensor:
            The updated tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, index=index, t_update=t_update)
    check_tensor_ipu_and_tile_set(t=t, index=index, t_update=t_update)

    settings = ctx._get_op_settings('dynamicupdate_inplace')
    # This ensures that `t` is created by calling `popops::createSliceableTensorFromSlice`
    # with `t_update`.
    # Does the user need control over this?
    settings.inferTensorMappingToFrom = {0: 2}
    opid = _ir.OperatorIdentifier("ai.graphcore", "DynamicUpdateInplace", 1,
                                  _ir.NumInputs(3, 3), 1)
    op = pb_g.createConnectedOp_DynamicUpdateInplaceOp(
        {
            0: t.id,
            1: index.id,
            2: t_update.id
        }, {0: g._create_tensor_id("dynamicupdateinplace_out")}, opid, axes,
        sizes, no_overlap, settings, t_update._pb_tensor.info)

    return Tensor._from_pb_tensor(op.outTensor(0))
