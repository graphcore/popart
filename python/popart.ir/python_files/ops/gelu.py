# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.globals import gcg
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ["gelu"]


def gelu(t: Tensor) -> Tensor:
    """
    Computes the Gelu activation on a Tensor.
    https://arxiv.org/abs/1606.08415

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
            Output tensor.
    """
    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, t)

    settings = _ir.Settings(pb_g, 'gelu')
    opid = _ir.OperatorIdentifier("ai.graphcore", "Gelu", 1, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_GeluOp(
        {0: t.id}, {0: g._create_tensor_id(f"gelu_out")}, opid, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
