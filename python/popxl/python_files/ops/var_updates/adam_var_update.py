# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Union

from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor

from ..utils import check_in_graph
from .utils import handle_optimizer_value


@op_debug_context
def adam_var_update(
        t: Tensor,
        x: Tensor,
        r1: Tensor,
        r2: Tensor,
        learning_rate: Optional[Union[float, Tensor]] = None,
        max_weight_norm: Optional[Union[float, Tensor]] = None) -> Tensor:
    """
     Calculate the updated weight tensor for Adam/LAMB.

     x = updater term (see :func:`~popxl.ops.adamupdater`)
     lr = learning rate
     max_weight_norm = max weight norm (c.f. phi or scaling function in Lamb paper)
     r1 = (Lamb) L2 norm of the weight (w)
     r2 = (Lamb) L2 norm of the updater term (x)

     Lamb r1 (FP32):
     r1 = ||w||_2                    (without Lamb or φ(r1) == 0: r1/r2 = 1)
       special case: replicated weight sharding; every replica only stores a
       shard of w, therefore the sum-of-squares is computed replicated, and
       thereafter all-reduced before every replica takes the square root of r1sq

     Lamb r2 (FP32):
     r2 = ||x||_2                    (without Lamb or r2 == 0: r1/r2 = 1)
       special case: replicated weight sharding; every replica only stores a
       shard of x, therefore the sum-of-squares is computed replicated, and
       thereafter all-reduced before every replica takes the square root of r2sq

     scale factor:
     φ(r1) = min(r1, max_weight_norm)

     variable update:
     w -= φ(r1) / r2 * lr * x
          ^^^^^^^^^^
          Lamb trust ratio

    Args:
        t (Tensor): The weight to update.
        x (Tensor): The updater term.
        r1 (Tensor): The r1 squared input tensor.
        r2 (Tensor): The r2 squared input tensor.
        learning_rate (Optional[Union[float, Tensor]]): Optional learning rate tensor to use. Will
        be constant if this argument is a float or None.
            Defaults to None.
        max_weight_norm (Optional[Union[float, Tensor]]): Optional max weight tensor to use. Will be
            constant if this argument is is a float or None.
            Defaults to None.

    Returns:
        Tensor: The updated weight tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, x=x, r1=r1, r2=r2)

    settings = ctx._get_op_settings('adam_var_update')

    ins = {0: t.id, 1: x.id, 2: r1.id, 3: r2.id}

    lr_ov = handle_optimizer_value(learning_rate, ins, 4)
    max_weight_norm_ov = handle_optimizer_value(max_weight_norm, ins, 5)

    op = pb_g.createConnectedOp_AdamVarUpdateOp(
        ins,
        {0: g._create_tensor_id("adam_var_update_out")},
        lr_ov,
        max_weight_norm_ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
