# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def relu(t: Tensor) -> Tensor:
    """
    Compute the ReLU activation of a tensor.

    For more details, refer to `Rectifier (neural networks) <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`__.

    See also `ONNX Relu <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu>`__.

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
            Output tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('relu')
    opid = _ir.OperatorIdentifier("ai.onnx", "Relu", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReluOp(
        {0: t.id}, {0: g._create_tensor_id("relu_out")}, opid, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def relu_(t: Tensor) -> Tensor:
    """
    Compute the ReLU activation of a tensor (in-place).

    For more details, refer to `Rectifier (neural networks) <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`__.

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
            t with the relu activation applied on it inplace.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('relu_inplace')
    op = pb_g.createConnectedOp_ReluInplaceOp(
        {0: t.id}, {0: g._create_tensor_id("relu_inplace_out")}, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
