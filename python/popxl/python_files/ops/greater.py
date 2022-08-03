# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def greater(input: Tensor, other: Tensor) -> Tensor:
    """
    Computes where the first tensor is greater than the second tensor.

    This is an element-wise operation (with NumPy-style broadcasting support).

    See also
    `Pytorch greater <https://pytorch.org/docs/stable/generated/torch.greater.html>`__,
    `NumPy greater <https://numpy.org/doc/stable/reference/generated/numpy.greater.html>`__.

    Args:
        input (Tensor):
            The first input operand for the logical operator.
        other (Tensor):
            The second input operand for the logical operator.
    Returns:
        Tensor:
            A tensor with `true` if the corresponding element of `input` is greater than `other` and `false` otherwise.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, input=input, other=other)
    check_tensor_ipu_and_tile_set(input=input, other=other)

    settings = ctx._get_op_settings("greater")
    opid = _ir.OperatorIdentifier("ai.onnx", "Greater", 9, _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_GreaterOp(
        {0: input.id, 1: other.id},
        {
            0: g._create_tensor_id("greater_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
