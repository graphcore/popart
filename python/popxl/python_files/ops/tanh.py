# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def tanh(t: Tensor) -> Tensor:
    """
    Compute the hyperbolic tangent function element-wise on a Tensor.

    This is similar to :onnxop:`Tanh`.

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

    settings = ctx._get_op_settings('tanh')
    opid = _ir.OperatorIdentifier("ai.onnx", "Tanh", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_TanhOp(
        {0: t.id}, {0: g._create_tensor_id("tanh_out")}, opid, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
