# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Union
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor

from ..utils import check_in_graph
from .utils import handle_optimizer_value

__all__ = [
    'accumulate_', 'accumulate_square_', 'accumulate_mean_',
    'accumulate_moving_average_', 'accumulate_moving_average_square_'
]


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

    check_in_graph(g, t, X)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(g, f, ins, 2)

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

    check_in_graph(g, t, X)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(g, f, ins, 2)

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
    ```
    with g:
        accum = pir.variable(0, dtype=pir.float32)
        a = pir.variable(1, dtype=pir.float32)
        b = pir.variable(2, dtype=pir.float32)
        accumulate_mean(accum, a, 0.0)
        accumulate_mean(accum, b, 1.0)
    ```
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

    check_in_graph(g, t, X, step)

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

    check_in_graph(g, t, X)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(g, f, ins, 2)

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

    check_in_graph(g, t, X)

    ins = {0: t.id, 1: X.id}

    ov = handle_optimizer_value(g, f, ins, 2)

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
