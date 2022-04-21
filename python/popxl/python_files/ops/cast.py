# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.dtypes import dtype
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def cast(t: Tensor, data_type: dtype) -> Tensor:
    """
    Cast a tensor to a specific data type.

    This operation casts tensor `t` to data type `data_type`.

    See also `ONNX Cast <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast>`__.

    Args:
        t (Tensor):
            The tensor to be cast.
        data_type (popxl.dtypes.dtype):
            The dtype to cast to.
    Returns:
        Tensor: The tensor cast to the specified type.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('cast')
    opid = _ir.OperatorIdentifier("ai.onnx", "Cast", 9, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_CastOp(
        {0: t.id},
        {0: g._create_tensor_id(f"{t.id}_{data_type._name}")},
        _to=data_type._pb_dtype,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
