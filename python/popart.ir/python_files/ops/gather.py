# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, convert_optional_float, check_tensor_ipu_and_tile_set


@op_debug_context
def gather(
        t: Tensor,
        indices: Tensor,
        axis: int = 0,
        available_memory_proportion: Optional[float] = None,
        zero_OOR=False,
) -> Tensor:
    """
    Selects multiple elements from an array.

    Elements are specified via `indices`, along a specified axis.
    Equivlent to `numpy.take`. Note that this is different from `torch.gather`.

    Examples:

    .. code-block:: python

        x = pir.variable(np.arange(16).reshape(4,4))
        # [[ 0,  1,  2,  3],
        #  [ 4,  5,  6,  7],
        #  [ 8,  9, 10, 11],
        #  [12, 13, 14, 15]]

        gather(x, [3, 1, 2]) == Tensor([x[3], x[1], x[2]])
        # [[12, 13, 14, 15],
        #  [ 4,  5,  6,  7],
        #  [ 8,  9, 10, 11]]

        gather(x, [[0,1], [1, 2]]) == gather(x, [0,1,1,2]).reshape(2, 2, 4)
        #  [[[ 0,  1,  2,  3],
        #    [ 4,  5,  6,  7]],
        #   [[ 4,  5,  6,  7],
        #    [ 8,  9, 10, 11]]]

    Args:
        t: Tensor
            Input tensor
        indices: Tensor
            The indices of the elements to extract
        axis: int
            Which axis to gather on. Default is 0.
        available_memory_proportion: Optional[float]
            The maximum proportion of available memory on each tile that this layer
            should consume temporarily during the course of the operation.
            Defaults to 1.0 if not set globally.
        zero_OOR: bool
            If False, out of range (OOR) indices will produce garbage data.
            If True, OOR indices will produce zeros.

    Returns:
        gather: Tensor
            The gathered elements concatenated.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, indices=indices)
    check_tensor_ipu_and_tile_set(t=t, indices=indices)

    available_memory_proportion = convert_optional_float(
        available_memory_proportion)

    opid = _ir.OperatorIdentifier("ai.onnx", "Gather", 11, _ir.NumInputs(2, 2),
                                  1)
    settings = ctx._get_op_settings("gather")
    op = pb_g.createConnectedOp_GatherOp(
        {
            0: t.id,
            1: indices.id
        }, {0: g._create_tensor_id("gather_out")},
        opid=opid,
        axis_=axis,
        available_memory_proportion_=available_memory_proportion,
        zeroOutOfRangeIndices_=zero_OOR,
        settings=settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def tied_gather(
        t: Tensor,
        indices: Tensor,
        axis: int = 0,
        available_memory_proportion: Optional[float] = None,
        zero_OOR=False,
) -> Tensor:
    """
    Select multiple elements from an array.

    Elements are specified given by `indices`, along a specified axis.
    Equivalent to `numpy.take`. Note that this is different from `torch.gather`.

    Numerically the same as the `gather` op but does not specify the tile
    layout of the `indices` tensor. When preceding a `matmul` op the tile
    layout of the indices is determined by the `matmul`, not the `tied_gather`.
    This has a has lower memory footprint but costs extra cycles due to the exchange.

    Examples:

    .. code-block:: python

        x = pir.variable(np.arange(16).reshape(4,4))
        # [[ 0,  1,  2,  3],
        #  [ 4,  5,  6,  7],
        #  [ 8,  9, 10, 11],
        #  [12, 13, 14, 15]]

        gather(x, [3, 1, 2]) == Tensor([x[3], x[1], x[2]])
        # [[12, 13, 14, 15],
        #  [ 4,  5,  6,  7],
        #  [ 8,  9, 10, 11]]

        gather(x, [[0,1], [1, 2]]) == gather(x, [0,1,1,2]).reshape(2, 2, 4)
        #  [[[ 0,  1,  2,  3],
        #    [ 4,  5,  6,  7]],
        #   [[ 4,  5,  6,  7],
        #    [ 8,  9, 10, 11]]]

    Args:
        t: Tensor
            Input tensor
        indices: Tensor
            The indices of the elements to extract
        axis: int
            Which axis to gather on. Default is 0.
        available_memory_proportion: Optional[float]
            The maximum proportion of available memory on each tile that this layer
            should consume temporarily during the course of the operation.
            Defaults to 1.0 if not set globally.
        zero_OOR: bool
            If False, out of range (OOR) indices will produce garbage data.
            If True, OOR indices will produce zeros.

    Returns:
        gather: Tensor
            The gathered elements concatenated.
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, indices=indices)
    check_tensor_ipu_and_tile_set(t=t, indices=indices)

    available_memory_proportion = convert_optional_float(
        available_memory_proportion)

    settings = ctx._get_op_settings("tiedgather")
    op = pb_g.createConnectedOp_TiedGatherOp(
        {
            0: t.id,
            1: indices.id
        }, {0: g._create_tensor_id("tiedgather_out")},
        axis_=axis,
        available_memory_proportion_=available_memory_proportion,
        zeroOutOfRangeIndices_=zero_OOR,
        settings=settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
