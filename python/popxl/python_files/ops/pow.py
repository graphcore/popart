# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import popxl
from typing import Union
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor, TensorLike
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def pow(t: Tensor, e: Union[float, int, TensorLike]) -> Tensor:
    """
    Raise the elements of `t` to the power of `e`.

    If `e` is `TensorLike`, then `t[i]` will be raised to the power of `e[i]`.
    If `e` is a `float` or `int`, all elements will be raised to the power of `e`.
    Follows NumPy broadcasting rules.

    Args:
        t (Tensor):
            Input tensor.
        e (Union[float, int, TensorLike]):
            Exponent tensor.
    Returns:
        Tensor:
            Output tensor containing the result of `t` raised to the power of `e`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    if not isinstance(e, Tensor):
        e = popxl.constant(e)

    tensors_to_check = dict(t=t, e=e)
    ins = {0: t.id, 1: e.id}

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    settings = ctx._get_op_settings("pow")
    opid = _ir.OperatorIdentifier("ai.onnx", "Pow", 7, _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_PowOp(
        ins, {0: g._create_tensor_id("pow_out")}, opid, settings
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
