# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def argmax(t: Tensor, dim: Optional[int] = 0,
           keepdim: Optional[bool] = False) -> Tensor:
    """
    Compute the argmax of a tensor.

    Compute the indices of the max elements of the input tensor's element along the provided axis.
    The resulting tensor has the same rank as the input if keepdim is True.
    If keepdim is False, then the resulting tensor has the reduced dimension pruned.

    See also
    `PyTorch Tensor.argmax <https://pytorch.org/docs/stable/generated/torch.Tensor.argmax.html>`__,
    `NumPy argmax <https://numpy.org/doc/stable/reference/generated/numpy.argmax.html>`__,
    `ONNX ArgMax <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax>`__.

    Args:
        t (Tensor):
            Input data tensor.
        dim (int):
            The axis in which to compute the arg indices.
        keepdim (bool):
            Keep the reduced dimension or not, True means keep reduced dimension.
    Returns:
        Tensor:
            The indices of the maximum values of a tensor across a dimension.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    settings = ctx._get_op_settings('argmax')
    opid = _ir.OperatorIdentifier("ai.onnx", "ArgMax", 11, _ir.NumInputs(1, 1),
                                  1)
    op = pb_g.createConnectedOp_ArgMaxOp({
        0: t.id,
    }, {
        0: g._create_tensor_id("argmax_out"),
    }, opid, dim, int(keepdim), settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def argmin(t: Tensor, dim: Optional[int] = 0,
           keepdim: Optional[bool] = False) -> Tensor:
    """
    Compute the argmin of a tensor.

    Compute the indices of the min elements of the input tensor's element along the provided axis.
    The resulting tensor has the same rank as the input if keepdim is True.
    If keepdim is False, then the resulting tensor has the reduced dimension pruned.

    See also
    `PyTorch Tensor.argmin <https://pytorch.org/docs/stable/generated/torch.Tensor.argmin.html>`__,
    `NumPy argmin <https://numpy.org/doc/stable/reference/generated/numpy.argmin.html>`__,
    `ONNX ArgMin <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin>`__.

    Args:
        t (Tensor):
            Input data tensor.
        dim (int):
            The axis in which to compute the arg indices.
        keepdim (bool):
            Keep the reduced dimension or not, True means keep reduced dimension.
    Returns:
        Tensor:
            The indices of the minimum values of a tensor across a dimension.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    settings = ctx._get_op_settings('argmin')
    opid = _ir.OperatorIdentifier("ai.onnx", "ArgMin", 11, _ir.NumInputs(1, 1),
                                  1)
    op = pb_g.createConnectedOp_ArgMinOp({
        0: t.id,
    }, {
        0: g._create_tensor_id("argmin_out"),
    }, opid, dim, int(keepdim), settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
