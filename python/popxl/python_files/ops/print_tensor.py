# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def print_tensor(t: Tensor,
                 title: str = None,
                 print_self: bool = True,
                 print_gradient: bool = False) -> Tensor:
    """
    Print a tensor.

    You need to use the output tensor of this op, otherwise the `removeIsolatedTensors` pattern
    will remove this op and it will not print the tensor.

    Args:
        t (Tensor): The tensor to print.
        title (str, optional): Title to print. Defaults to None.
        print_self (bool, optional): Print the tensor itself. Defaults to `True`.
        print_gradient (bool, optional): Indicates if the associated gradient tensor of t is also printed (`True`) or not (`False`).
            Defaults to False.

    Returns:
        Tensor: The input tensor, unchanged.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('print_tensor')
    opid = _ir.OperatorIdentifier("ai.graphcore", "PrintTensor", 1,
                                  _ir.NumInputs(1, 1), 1)
    if title is None:
        title = f"print{t.name}"

    op = pb_g.createConnectedOp_PrintTensorOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("print_out"),
        },
        opid,
        print_self,
        print_gradient,
        title,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
