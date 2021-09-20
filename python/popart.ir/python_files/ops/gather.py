# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
import popart._internal.ir as _ir
from popart.ir.globals import gcg
from popart.ir.tensor import Tensor
from .utils import check_in_graph, convert_optional_float

__all__ = ["gather", "tied_gather"]


def gather(t: Tensor,
           indices: Tensor,
           axis: int = 0,
           available_memory_proportion: Optional[float] = None) -> Tensor:
    """
    Select multiple elements from an array, given by `indices`, along a specified axis.

    When `axis == 0`, it is equivlent to numpy "fancy indexing".

    Pseudo example:
    ```
    gather(x, [1, 2, 3]) == [x[3], x[7], x[2]]
    ```

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
                
    Returns:
        gather: Tensor
            The gathered elements concatenated.
    """
    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, t)
    check_in_graph(g, indices)

    available_memory_proportion = convert_optional_float(
        available_memory_proportion)

    opid = _ir.OperatorIdentifier("ai.graphcore", "Gather", 11,
                                  _ir.NumInputs(2, 2), 1)
    settings = _ir.Settings(pb_g, "gather")
    op = pb_g.createConnectedOp_GatherOp(
        {
            0: t.id,
            1: indices.id
        }, {0: g._create_tensor_id("gather_out")},
        opid=opid,
        axis_=axis,
        available_memory_proportion_=available_memory_proportion,
        settings=settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


def tied_gather(t: Tensor,
                indices: Tensor,
                axis: int = 0,
                available_memory_proportion: Optional[float] = None) -> Tensor:
    """
    Select multiple elements from an array, given by `indices`, along a specified axis.

    When `axis == 0`, it is equivlent to numpy "fancy indexing".

    Numerically the same as the `gather` op but does not specify the tile
    layout of the `indices` tensor. When preceding a `matmul` op the tile
    layout of the indices is determined by the `matmul`, not the `tied_gather`.
    This has a has lower memory footprint but costs extra cycles due to the exchange.

    Pseudo example:
    ```
    tied_gather(x, [1, 2, 3]) == [x[3], x[7], x[2]]
    ```

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
                
    Returns:
        gather: Tensor
            The gathered elements concatenated.
    """

    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, t)
    check_in_graph(g, indices)

    available_memory_proportion = convert_optional_float(
        available_memory_proportion)

    settings = _ir.Settings(pb_g, "tiedgather")
    op = pb_g.createConnectedOp_TiedGatherOp(
        {
            0: t.id,
            1: indices.id
        }, {0: g._create_tensor_id("tiedgather_out")},
        axis_=axis,
        available_memory_proportion_=available_memory_proportion,
        settings=settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
