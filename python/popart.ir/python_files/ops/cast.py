# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.dtypes import dtype
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['cast']


@op_debug_context
def cast(t: Tensor, data_type: dtype) -> Tensor:
    """
    Casts tensor `t` to data type `dtype`.

    Args:
        t: Tensor
            Tensors to be casted.
        data_type: popart.ir.dtypes.dtype
            Dtype to cast to
    Returns:
        add: Tensor
            The sum of lhs and rhs
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

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
