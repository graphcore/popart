# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def mean(t: Tensor,
         axis: Optional[Union[int, Iterable[int]]] = None,
         keepdims: bool = False) -> Tensor:
    """
    Compute the arithmetic mean of elements in a tensor along axes.

    See also `PyTorch Tensor.mean <https://pytorch.org/docs/stable/generated/torch.Tensor.mean.html>`__, `NumPy mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`__, `ONNX Mean <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean>`__.

    Args:
        t (Tensor):
            Tensor to compute mean.
        axis (int or list):
            Axis or axes to computer the mean along. If none is provided all axes will
            be reduced. If axis is negative it counts from the
            last to the first axis.
        keepdims (bool):
            Keep the axis that is being reduced (`True` or not (`False`).

    Returns:
        Tensor
            The reduced tensor containing the arithmetic means computed along the specified axes.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings('ReduceMean')
    opid = _ir.OperatorIdentifier("ai.onnx", "ReduceMean", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReduceMeanOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("mean_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
