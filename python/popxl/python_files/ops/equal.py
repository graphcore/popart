# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def equal(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Apply an elementwise equality operation.

    Follows NumPy broadcasting rules.

    See also `PyTorch Tensor.equal <https://pytorch.org/docs/stable/generated/torch.Tensor.equal.html>`__, `NumPy equal <https://numpy.org/doc/stable/reference/generated/numpy.equal.html>`__, `ONNX Equal <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal>`__.

    Args:
        lhs, rhs: Tensor
            Tensors to be compared.
    Returns:
        Tensor:
            The value (lhs == rhs)
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    settings = ctx._get_op_settings('equal')
    opid = _ir.OperatorIdentifier("ai.onnx", "Equal", 7, _ir.NumInputs(2, 2),
                                  1)
    op = pb_g.createConnectedOp_EqualOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("equal_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
