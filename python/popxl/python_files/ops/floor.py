# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def floor(t: Tensor) -> Tensor:
    """
    Compute the floor of the elements of input tensor. NaN values are propagated.

    See also `PyTorch torch.floor <https://pytorch.org/docs/stable/generated/torch.floor.html#torch.floor>`__, `NumPy floor <https://numpy.org/doc/stable/reference/generated/numpy.floor.html>`__, `ONNX Floor <https://github.com/onnx/onnx/blob/main/docs/Operators.md#floor>`__.

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

    settings = ctx._get_op_settings("floor")
    opid = _ir.OperatorIdentifier("ai.onnx", "Floor", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_FloorOp(
        {0: t.id}, {0: g._create_tensor_id("floor_out")}, opid, settings
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
