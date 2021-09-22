# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.globals import gcg
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['div']


def div(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Divides two Tensors element-wise.
    Follows numpy broadcasting rules. Arguments must have the same dtype.
    Output will be the same dtype as the inputs.
    With integer values floor division is used.

    Args:
        x1, x2: Tensor
            Tensors to be divided.
    Returns:
        mul: Tensor
            The division of x1 by x2
    """
    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, x1, x2)

    settings = _ir.Settings(pb_g, 'div')
    opid = _ir.OperatorIdentifier("ai.onnx", "Div", 7, _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_DivOp(
        {
            0: x1.id,
            1: x2.id
        },
        {
            0: g._create_tensor_id("div_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
