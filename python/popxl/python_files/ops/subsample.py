# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Iterable, Union
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def subsample(t: Tensor, strides: Union[int, Iterable[int]]) -> Tensor:
    """
    Subsamples a tensor by selecting every n'th element from each dimension.
    The subsample count N is provided for each dimension.

    Args:
        t (Tensor):
            The input tensor to subsample.
        strides (Union[int, Iterable[int]]):
            A list of strides for each dimension of the input tensor
    Returns:
        Tensor:
            A subsampled output tensor.
    Raises:
        ValueError: Thrown if the length of the strides list is different
            to the rank of the input tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    if isinstance(strides, int):
        strides = [strides]

    if len(strides) != t.rank:
        raise ValueError(
            "The length of subsample `strides` must match the rank of the input tensor."
        )

    settings = ctx._get_op_settings("subsample")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "Subsample", 1, _ir.NumInputs(1, 1), 1
    )
    op = pb_g.createConnectedOp_SubsampleOp(
        {0: t.id},
        {0: g._create_tensor_id("subsample_out")},
        strides_=strides,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
