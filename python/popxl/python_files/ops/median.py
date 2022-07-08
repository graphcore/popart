# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def median(
    t: Tensor, axis: Optional[Union[int, Iterable[int]]] = None, keepdims: bool = False
) -> Tensor:
    """
    Compute the median of elements in a tensor along axes.

    See also `PyTorch Tensor.median <https://pytorch.org/docs/stable/generated/torch.Tensor.median.html>`__, `NumPy median <https://numpy.org/doc/stable/reference/generated/numpy.median.html>`__.

    Args:
        t (Tensor):
            Tensor to compute median of.
        axis (int or list):
            Axis or axes to compute the median along. If none is provided all axes will
            be reduced. If axis is negative it indexes from the
            last to the first axis.
        keepdims (bool):
            Keep the axis that is being reduced (`True`) or not (`False`).

    Returns:
        Tensor:
            The reduced tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings("ReduceMedian")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "ReduceMedian", 1, _ir.NumInputs(1, 1), 1
    )
    op = pb_g.createConnectedOp_ReduceMedianOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("median_out"),
            1: g._create_tensor_id("median_indices_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
