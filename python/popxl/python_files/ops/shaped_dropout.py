# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set
from typing import Iterable


@op_debug_context
def shaped_dropout(
    t: Tensor, seed_tensor: Tensor, shape: Iterable[int], ratio: float
) -> Tensor:
    """
    Add a shaped dropout operation to the input tensor.

    Applies a shaped dropout to the input tensor `t`. This operator requires a `shape`
    parameter that is used to define the shape of the dropout mask so that strongly
    correlated features in the input tensor `t` can be preserved. The `shape` parameter
    must be broadcastable to the input tensor `t`. The dropout mask is created using
    samples from a Bernoulli distribution seeded with a seed tensor `seed_tensor`.

    Args:
        t (Tensor):
            The Tensor to apply the shaped dropout operation to.
        seed_tensor (Tensor):
            The Tensor used to seed the probability distribution which generates the dropout mask.
            Must have data type uint32 and shape [2,].
        shape (Iterable[int]):
            The shape of the dropout mask. This must be broadcastable to the input tensor.
        ratio (float):
            The probability of dropping an input feature. Default = 0.5.

    Returns:
        Tensor: A new tensor with the shaped dropout applied.
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, seed_tensor=seed_tensor)
    check_tensor_ipu_and_tile_set(t=t, seed_tensor=seed_tensor)

    settings = ctx._get_op_settings("shapeddropout")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "ShapedDropout", 1, _ir.NumInputs(1, 1), 1
    )
    op = pb_g.createConnectedOp_ShapedDropoutOp(
        {0: t.id, 1: seed_tensor.id},
        {0: g._create_tensor_id("shapeddropout_out")},
        opid,
        ratio,
        shape,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
