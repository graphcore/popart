# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def sign(t: Tensor) -> Tensor:
    """
    Return the sign of each element in the Tensor (-1, 0 or 1). NaN values have a sign of 0.

    Args:
        t (Tensor):
            Input Tensor.

    Returns:
        Tensor: element-wise sign of input.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("sign")
    opid = _ir.OperatorIdentifier("ai.onnx", "Sign", 9, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_SignOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("sign_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
