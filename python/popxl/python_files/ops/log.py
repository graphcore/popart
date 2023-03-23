# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def log(t: Tensor) -> Tensor:
    """
    Compute the log of the elements of input tensor.

    See also `PyTorch torch.log <https://pytorch.org/docs/stable/generated/torch.log.html#torch.log>`__, `NumPy log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`__, `ONNX Log <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log>`__.

    Args:
        t (Tensor):
            Input tensor.
    Returns:
        Tensor:
            Output tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("log")
    opid = _ir.OperatorIdentifier("ai.onnx", "Log", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_LogOp(
        {0: t.id}, {0: g._create_tensor_id("log_out")}, opid, settings
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def log2(t: Tensor) -> Tensor:
    """
    Compute the base-2 logarithm of input tensor.
    Args:
        t (Tensor):
            Input tensor.
    Returns:
        Tensor:
            Output tensor.
    """

    return log(t) / log(popxl.constant(2, t.dtype))
