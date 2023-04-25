# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def identity(t: Tensor, output_name: Optional[str] = None) -> Tensor:
    """
    Input is equal to the output. This can also be used to rename a Tensor.

    Args:
        t (Tensor):
            Tensor to provide as an output.
        output_name (str):
            Name of output tensor

    Returns:
        Tensor: equal to input.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    output_name = output_name if output_name else f"identity_{t.name}"

    settings = ctx._get_op_settings("identity")
    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_IdentityOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id(output_name),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


# Alias
rename = identity
