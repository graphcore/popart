# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def negate(t: Tensor) -> Tensor:
    """
    Perform elementwise negation (two's complement) of a tensor.

    Args:
        t (Tensor): The input tensor.
    Returns:
        Tensor:
            The output tensor that is the elementwise negation of the input.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('negate')
    opid = _ir.OperatorIdentifier("ai.onnx", "Neg", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_NegateOp(
        {0: t.id}, {0: g._create_tensor_id("negate_out")}, opid, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
