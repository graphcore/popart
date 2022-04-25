# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union
import popart
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor, constant
from popxl.dtypes import float32
from .utils import check_in_graph, convert_optional_int, check_tensor_ipu_and_tile_set


@op_debug_context
def nll_loss_with_softmax_grad(
        probs: Tensor,
        labels: Tensor,
        loss_grad: Union[float, Tensor] = 1,
        reduction: popart.ReductionType = popart.ReductionType.Mean,
        ignore_index: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """ Compute the negative log likelihood loss.

        Compute the negative log likelihood loss `l` and returns the gradient `dE/dx` where `probs = softmax(x)`.
        `loss_grad` should be the gradient `dE/dl`, where `E` is the error from which back propagation is initialised.
        Typically, `E = l` therefore in order to return `dl/dx` the `loss_grad` should be `dl/dl` which would be `1`.

    Args:
        probs (Tensor): The probabilities. Expected to be the output of :py:func:`~popxl.ops.softmax`.
        labels (Tensor): The labels. Target values for the probabilities.
        loss_grad (Tensor): The gradient, `dE/dl`. Supports float32 dtypes with float16 `probs`
        reduction (ReductionType): Specify how to reduce the loss. Defaults to `ReductionType.Mean`.
        ignore_index (Optional[int]): Specify label values that should not contribute to `l` or `dE/dx`. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: A tuple of the loss and the gradient: (`l`, `dE/dx`).
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    if not isinstance(loss_grad, Tensor):
        loss_grad = constant(loss_grad, float32)

    check_in_graph(g, probs=probs, labels=labels, loss_grad=loss_grad)
    check_tensor_ipu_and_tile_set(probs=probs,
                                  labels=labels,
                                  loss_grad=loss_grad)

    settings = ctx._get_op_settings('nll_loss_with_softmax_grad')
    op = pb_g.createConnectedOp_NlllWithSoftmaxGradDirectOp(
        {
            0: probs.id,
            1: labels.id,
            2: loss_grad.id
        },
        {
            0: g._create_tensor_id("loss"),
            1: g._create_tensor_id("dx"),
        },
        convert_optional_int(ignore_index),
        reduction,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0)), Tensor._from_pb_tensor(
        op.outTensor(1))
