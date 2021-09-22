# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.dtypes import dtype
from popart.ir.globals import gcg
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['cast']


def cast(t: Tensor, data_type: dtype) -> Tensor:
    """
    Adds two Tensors element-wise.
    Follows numpy broadcasting rules.
    Arguments must have the same dtype.

    Args:
        lhs, rhs: Tensor
            Tensors to be added.
    Returns:
        add: Tensor
            The sum of lhs and rhs
    """
    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, t)

    settings = _ir.Settings(pb_g, 'cast')
    opid = _ir.OperatorIdentifier("ai.onnx", "Cast", 9, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_CastOp(
        {0: t.id},
        {0: g._create_tensor_id(f"{t.id}_{data_type._name}")},
        _to=data_type._pb_dtype,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
