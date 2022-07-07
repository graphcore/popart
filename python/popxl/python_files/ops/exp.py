# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def exp(t: Tensor) -> Tensor:
    """
    Compute the exponential of the elements of input tensor.

    See also `PyTorch Tensor.exp <https://pytorch.org/docs/stable/generated/torch.Tensor.exp.html>`__, `NumPy exp <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`__, `ONNX Exp <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp>`__.

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

    settings = ctx._get_op_settings('exp')
    opid = _ir.OperatorIdentifier("ai.onnx", "Exp", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ExpOp(
        {0: t.id}, {0: g._create_tensor_id("exp_out")}, opid, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def exp_(t: Tensor) -> Tensor:
    """
    Compute the exponential of the elements of input tensor (in-place).

    See also `PyTorch Tensor.exp_ <https://pytorch.org/docs/stable/generated/torch.Tensor.exp_.html>`__.

    Args:
        t (Tensor):
            Input tensor.
    Returns:
        Tensor:
            The input tensor with the exp applied to it in place.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('exp_inplace')
    op = pb_g.createConnectedOp_ExpInplaceOp(
        {0: t.id}, {0: g._create_tensor_id("exp_inplace_out")}, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
