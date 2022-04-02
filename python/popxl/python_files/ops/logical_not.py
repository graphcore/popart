# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl import dtypes
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, cast_if_needed


@op_debug_context
def logical_not(t: Tensor) -> Tensor:
    """
    Compute the element-wise NOT of a tensor.

    Inputs will be cast to bool if needed.

    This is similar to :onnxop:`Not`.

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    t = cast_if_needed(t, dtypes.bool)

    settings = ctx._get_op_settings('not')
    opid = _ir.OperatorIdentifier("ai.onnx", "Not", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_NotOp(
        {0: t.id},
        {0: g._create_tensor_id("not_out")},
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
