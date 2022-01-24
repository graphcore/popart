# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def dropout(t: Tensor, seed_tensor: Tensor, p: float):
    """
    Randomly zero elements of the input tensor.

    This operation will zero elements of tensor `t` with a probability of `p`.
    The dropout mask is created using samples from a Bernoulli distribution
    seeded with the `seed_tensor`.

    The user needs to manage updating the `seed_tensor` for each forward pass and replica.

    Args:
        t (Tensor):
            Tensor for drop out to be applied.
        seed_tensor (Tensor):
            Used to seed the probability distribution which generates the dropout mask. Must have data type uint32 and shape (2,).
        p (float):
            Probability an element will be zeroed

    Returns:
        Tensor
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, seed_tensor=seed_tensor)
    check_tensor_ipu_and_tile_set(t=t, seed_tensor=seed_tensor)

    settings = ctx._get_op_settings('dropout')
    opid = _ir.OperatorIdentifier("ai.onnx", "Dropout", 10, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_DropoutOp(
        {
            0: t.id,
            1: seed_tensor.id
        },
        {0: g._create_tensor_id("dropout_out")},
        ratio_=p,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
