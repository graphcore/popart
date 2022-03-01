# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Union
import popart._internal.ir as _ir
from popxl import dtypes
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor

from ..utils import check_in_graph, check_tensor_ipu_and_tile_set
from .utils import handle_optimizer_value


@op_debug_context
def accumulate_(t: Tensor, X: Tensor,
                f: Optional[Union[float, Tensor]] = None) -> Tensor:
    """
    Updates tensor `t` inplace using `t = t + (f * X)`.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    `t` and `X` must be the same shape, but can be different types.
    `f` must be scalar.

    Args:
        t: Tensor
            Tensor to be updated.
        X: Tensor
            Value to update the variable
        f: Optional[Union[float, Tensor]]
            Optional scalar to apply to update before the addition.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(t=t, X=X)
    if isinstance(f, Tensor):
        tensors_to_check['f'] = f

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(f, ins, 2)

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        ins,
        {
            0: g._create_tensor_id('accumulate__' + t.name),
        },
        _ir.AccumulationType.DampenedAdd
        if f is not None else _ir.AccumulationType.Add,
        ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def accumulate_square_(t: Tensor, X: Tensor,
                       f: Union[float, Tensor] = 1.0) -> Tensor:
    """
    Updates tensor `t` inplace using `t = t + (f * X^2)`.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    `t` and `X` must be the same shape, but can be different types.
    `f` must be scalar.

    Args:
        t: Tensor
            Tensor to be updated.
        X: Tensor
            Value to update the variable
        f: Optional[Union[float, Tensor]]
            Optional scalar to apply to update before the addition.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(t=t, X=X)
    if isinstance(f, Tensor):
        tensors_to_check['f'] = f

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(f, ins, 2)

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        ins,
        {
            0: g._create_tensor_id('accumulate_square__' + t.name),
        },
        _ir.AccumulationType.DampenedAddSquare,
        ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def accumulate_mean_(t: Tensor, X: Tensor,
                     step: Union[float, Tensor]) -> Tensor:
    """
    Updates a tensor `t` inplace using `t = (step/(step+1)) * t + (1/(step+1)) * X`.
    Intended to be used to keep track of the mean of a series of values.

    For example:

    .. code-block:: python

        with g:
            accum = popxl.variable(0, dtype=popxl.float32)
            a = popxl.variable(1, dtype=popxl.float32)
            b = popxl.variable(2, dtype=popxl.float32)
            accumulate_mean(accum, a, 0.0)
            accumulate_mean(accum, b, 1.0)

    will result with `accum` having the value `(a+b)/2 = 1.5`.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    `t` and `X` must be the same shape, but can be different types.
    `step` must be scalar.

    Args:
        `t`: Tensor
            Tensor to be updated.
        `X`: Tensor
            Value to update the variable
        step: Union[float, Tensor]]
            Value representing the number of previously accumulated values.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    step = t._ensure_tensor(step, dtype=dtypes.float32)

    check_in_graph(g, t=t, X=X, step=step)
    check_tensor_ipu_and_tile_set(t=t, X=X, step=step)

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        {
            0: t.id,
            1: X.id,
            2: step.id
        },
        {
            0: g._create_tensor_id('accumulate_mean__' + t.name),
        },
        _ir.AccumulationType.Mean,
        _ir.OptimizerValue(),
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def accumulate_moving_average_(t: Tensor, X: Tensor,
                               f: Union[float, Tensor]) -> Tensor:
    """
    Updates tensor `t` inplace using `t = (f * t) + ((1-f) * X)`.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    `t` and `X` must be the same shape, but can be different types.
    `f` must be scalar.

    Args:
        t: Tensor
            Tensor to be updated.
        X: Tensor
            Value to update the variable
        f: Union[float, Tensor]
            Scalar to apply to update before the addition.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(t=t, X=X)
    if isinstance(f, Tensor):
        tensors_to_check['f'] = f

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(f, ins, 2)

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        ins,
        {
            0: g._create_tensor_id('accumulate_moving_avg__' + t.name),
        },
        _ir.AccumulationType.MovingAverage,
        ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def accumulate_moving_average_square_(t: Tensor, X: Tensor,
                                      f: Union[float, Tensor]) -> Tensor:
    """
    Updates tensor `t` inplace using `t = (f * t) + ((1-f) * X^2)`.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    `t` and `X` must be the same shape, but can be different types.
    `f` must be scalar.

    Args:
        t: Tensor
            Tensor to be updated.
        X: Tensor
            Value to update the variable
        f: Union[float, Tensor]
            Scalar to apply to update before the addition.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(t=t, X=X)
    if isinstance(f, Tensor):
        tensors_to_check['f'] = f

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(f, ins, 2)

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        ins,
        {
            0: g._create_tensor_id('accumulate_moving_avg_square__' + t.name),
        },
        _ir.AccumulationType.MovingAverageSquare,
        ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def accumulator_scale_(t: Tensor, f: Union[float, Tensor]) -> Tensor:
    """
    Inplace multiplies tensor 't' by a factor `f`: t = t * f.

    Will directly zero the input tensor if the factor is const and 0.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.

    Args:
        t: Tensor
            Tensor to be updated.
        f: Union[float, Tensor]
            The scalar to multiply by. If a float this will be a const multiplication, otherwise
            will multiply by the values of the (non-const) tensor `f` elementwise.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(t=t)
    if isinstance(f, Tensor):
        tensors_to_check['f'] = f

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    ins = {0: t.id}

    ov = handle_optimizer_value(f, ins, 2)

    settings = ctx._get_op_settings('accumulator_scale')
    op = pb_g.createConnectedOp_AccumulatorScaleOp(
        ins,
        {
            0: g._create_tensor_id('accumulator_scale__' + t.name),
        },
        ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def accumulator_zero_(t: Tensor) -> Tensor:
    """
    An AccumulatorScaleOp with a factor of 0, so zeroes the input tensor.

    Args:
        t: Tensor
            Tensor to be zeroed.
    Returns:
        updated: Tensor
            An alias to the input.
    """
    return accumulator_scale_(t, 0.0)


def sparse_accumulate_(t: Tensor,
                       X: Tensor,
                       indices: Tensor,
                       axis: int = 0,
                       f: Optional[Union[float, Tensor]] = None,
                       W: Optional[Tensor] = None) -> Tensor:
    """
    Applies a sparse accumulate operation to t.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    `t` and `X` must be the same shape, but can be different types.

    Detail:

    Say you have:
    w -> Gather -> x

    In backward pass you have:
    dW <- GatherGrad <- x

    and when the optimiser step is grown:
    dW <- GatherGrad <- x
     \
      Accumulate -> accum'
     /
    accum

    GatherGrad is essentially a scatter. Then we Accumulate the resultant dW on
    accum. This involves creating an extra dW tensor, so instead we can do:

                  x
                  |
                  V
    accum -> SparseAccumulate -> accum'

    Where SparseAccumulate can in one operation, without extra space, accumulate
    the slices of x into accum as required.

    ---------

    The input tensor W is an optional input. This is can be used when two different views of the
    weight are consumed in the forward pass, and one of those ops is a Gather, thus requiring a
    SparseAccumulate in the weight update step.

    We connect the op to the other view of the weight than the one this SparseAccumulate is for.
    Then, the lowering will clone that tensor (and its layout) when creating accum.

    Args:
        t (Tensor): Tensor to be updated
        X (Tensor): Value to update the variable.
        indices (Tensor): The indices of the scatter operation.
        axis (int, optional): Which axis to set on. Default is 0.
        f (Optional[Union[float, Tensor]], optional): Optional scalar to apply to update before the
            addition. Defaults to None.
        W (Optional[Tensor], optional):  Tile mapping reference tensor for `t` to be cloned from.


    Returns:
        Tensor: An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(t=t, X=X, indices=indices)
    ins = {0: t.id, 1: X.id, 3: indices.id}

    ov: _ir.OptimizerValue
    if isinstance(f, Tensor):
        tensors_to_check['f'] = f
        ins[2] = f.id
        ov = _ir.OptimizerValue(0.0, False)
    elif f is None:
        ov = _ir.OptimizerValue()
    else:
        ov = _ir.OptimizerValue(f)

    if W is not None:
        tensors_to_check['W'] = W
        ins[4] = W.id

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    settings = ctx._get_op_settings('sparse_accumulate')
    op = pb_g.createConnectedOp_SparseAccumulateOp(
        ins,
        {
            0: g._create_tensor_id('sparse_accumulate__' + t.name),
        },
        _ir.AccumulationType.DampenedAdd
        if f is not None else _ir.AccumulationType.Add,
        ov,
        axis,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
