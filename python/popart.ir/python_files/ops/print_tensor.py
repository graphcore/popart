# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['print_tensor']


@op_debug_context
def print_tensor(t: Tensor,
                 title: str = None,
                 print_self: bool = True,
                 print_gradient: bool = False) -> Tensor:
    """Print a tensor everytime this op runs in the graph. Note this will print in the context it
        is placed. E.g. if within a loop op, it will run each loop iteration.

    Args:
        t (Tensor): The tensor to print.
        title (str, optional): Title to print. Defaults to None.
        print_self (bool, optional): Print the tensor itself. Defaults to True.
        print_gradient (bool, optional): Whether to print the associated gradient tensor of t.
            Defaults to False.

    Returns:
        Tensor: The same unaltered tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

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
