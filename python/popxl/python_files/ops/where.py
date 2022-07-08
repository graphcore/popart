# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl import dtypes
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set, cast_if_needed


@op_debug_context
def where(condition: Tensor, lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Elementwise selection based on satisfying a condition.

    Choose elements from `lhs` or `rhs` depending on whether the corresponding element in `condition` is satisfied or not.
    The operator supports multi-directional broadcasting (NumPy style).

    See also `PyTorch Tensor.where <https://pytorch.org/docs/stable/generated/torch.Tensor.where.html>`__, `NumPy where <https://numpy.org/doc/stable/reference/generated/numpy.where.html>`__, `ONNX Where <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where>`__.

    Args:
        condition (Tensor):
            A boolean tensor where `True` indicates the `lhs` element and `False` the `rhs` element.
            The tensor will be cast to a bool if necessary.
        lhs (Tensor):
            The left hand side operand.
        rhs (Tensor):
            The right hand side operand.

    Returns:
        Tensor: The tensor containing elementwise `lhs if condition else rhs`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, condition=condition, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(condition=condition, lhs=lhs, rhs=rhs)

    condition = cast_if_needed(condition, dtypes.bool)

    settings = ctx._get_op_settings("where")
    opid = _ir.OperatorIdentifier("ai.onnx", "Where", 9, _ir.NumInputs(3, 3), 1)
    op = pb_g.createConnectedOp_WhereOp(
        {
            0: condition.id,
            1: lhs.id,
            2: rhs.id,
        },
        {
            0: g._create_tensor_id("where_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
