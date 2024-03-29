# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_int64_list
from typing import Optional, Iterable, Union


@op_debug_context
def logsumexp(
    t: Tensor, axis: Optional[Union[int, Iterable[int]]] = None, keepdims: bool = False
) -> Tensor:
    """
    Compute the log of the summed exponentials of elements in a tensor, along specified axes.

    Supported dtypes: floats.

    See also `PyTorch Tensor.logsumexp <https://pytorch.org/docs/stable/generated/torch.Tensor.logsumexp.html>`__.

    Args:
        t (Tensor):
            Tensor to compute the log of the summed exponentials of the elements.
        axis (int or list):
            Axis or axes to compute the log of the summed exponentials along. If none is specified all axes will
            be reduced. If axis is negative it indexes from the
            last to the first axis.
        keepdims (bool):
            Keep the axis that is being computed (`True`) or not (`False`).

    Returns:
        Tensor:
            A new tensor containing the log of the summed exponentials of the elements along the specified axes.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if isinstance(axis, int):
        axis = [axis]

    axis = convert_optional_int64_list(axis)

    settings = ctx._get_op_settings("ReduceLogSumExp")
    opid = _ir.OperatorIdentifier(
        "ai.onnx", "ReduceLogSumExp", 1, _ir.NumInputs(1, 1), 1
    )
    op = pb_g.createConnectedOp_ReduceLogSumExpOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("logsumexp_out"),
        },
        axes=axis,
        keepdims=keepdims,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
