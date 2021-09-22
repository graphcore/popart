# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.globals import gcg
from popart.ir.tensor import Tensor
from .utils import check_in_graph, cast_if_needed

__all__ = ['logical_not']


def logical_not(x: Tensor) -> Tensor:
    """
    Computes element-wise the value of NOT x.
    Inputs will be cast to bool if needed.

    Args:
        lhs, rhs: Tensor
            Tensors to be compared.
    Returns:
        out: Tensor
            The value (lhs NOT rhs)
    """
    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, x)

    x = cast_if_needed(x, dtypes.bool)

    settings = _ir.Settings(pb_g, 'not')
    opid = _ir.OperatorIdentifier("ai.onnx", "Not", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_NotOp(
        {0: x.id},
        {0: g._create_tensor_id("not_out")},
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
