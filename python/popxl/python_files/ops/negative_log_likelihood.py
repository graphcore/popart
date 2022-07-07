# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union
from typing_extensions import Literal

import popart._internal.ir as _ir
import popart

from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor, constant
from popxl.dtypes import float32
from .utils import check_in_graph, convert_optional_int, check_tensor_ipu_and_tile_set

REDUCTION_MAP = {
    'mean': popart.ReductionType.Mean,
    'sum': popart.ReductionType.Sum,
    'none': popart.ReductionType.NoReduction,
}

REDUCTION_TYPE = Literal['mean', 'sum', 'none']


def _to_reduction_enum(reduction: REDUCTION_TYPE) -> popart.ReductionType:
    try:
        return REDUCTION_MAP[reduction]
    except KeyError:
        raise ValueError(
            f"Not a valid reduction: {reduction}. "
            f"Must choose from: {', '.join(REDUCTION_MAP.keys())}") from None


@op_debug_context
def nll_loss(
        probs: Tensor,
        labels: Tensor,
        ignore_index: Optional[int] = None,
        reduction: REDUCTION_TYPE = 'mean',
        log_prob: bool = False,
) -> Tensor:
    """Compute the negative log likelihood loss.

    Compute the negative log likelihood loss `l` where `probs = softmax(x)`.
    The returned loss will be reduced by `reduction` (default mean) across items in `targets`.
    Any item in `target` equal to `ignore_index` will not contribute to `l` or `dl/dx`.

    See also `PyTorch nll_loss <https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss>`__, `ONNX NegativeLogLikelihoodLoss <https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss>`__.

    Args:
        probs (Tensor): The probabilities. Expected to be the output of :py:func:`~popxl.ops.softmax`.
        labels (Tensor): The labels. Target values for the probabilities.
        ignore_index (Optional[int], optional): Specify label values that should not contribute to the loss
        reduction (str): Specify how to reduce the loss. Defaults to `mean`. Options `mean`, `sum` and `none`
        log_prob (bool): If true input probabilities are logged

    Returns:
        Tensor:
            The calculated negative log likelihood loss.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, probs=probs, labels=labels)
    check_tensor_ipu_and_tile_set(probs=probs, labels=labels)

    reduction = _to_reduction_enum(reduction)

    settings = ctx._get_op_settings('nll')
    opid = _ir.OperatorIdentifier("ai.graphcore", "Nll", 1, _ir.NumInputs(
        2, 2), 1)
    op = pb_g.createConnectedOp_NllOp(
        {
            0: probs.id,
            1: labels.id,
        },
        {
            0: g._create_tensor_id("loss"),
        },
        opid=opid,
        ignoreIndex=convert_optional_int(ignore_index),
        reduction=reduction,
        inputIsLogProbability=log_prob,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def nll_loss_with_softmax_grad(
        probs: Tensor,
        labels: Tensor,
        loss_grad: Union[float, Tensor] = 1,
        ignore_index: Optional[int] = None,
        reduction: REDUCTION_TYPE = 'mean') -> Tuple[Tensor, Tensor]:
    """Compute the negative log likelihood loss.

    Compute the negative log likelihood loss `l` and returns the gradient `dE/dx` where `probs = softmax(x)`.
    `loss_grad` should be the gradient `dE/dl`, where `E` is the error from which back propagation is initialised.
    Typically, `E = l` therefore in order to return `dl/dx` the `loss_grad` should be `dl/dl` which would be `1`.

    Args:
        probs (Tensor): The probabilities. Expected to be the output of :py:func:`~popxl.ops.softmax`.
        labels (Tensor): The labels. Target values for the probabilities.
        loss_grad (Tensor): The gradient, `dE/dl`. Supports float32 dtypes with float16 `probs`
        reduction (ReductionType): Specify how to reduce the loss. Defaults to `mean`. Options `mean`, `sum` and `none`
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

    reduction = _to_reduction_enum(reduction)

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
