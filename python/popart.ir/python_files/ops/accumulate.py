# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Union
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor

from .utils import check_in_graph

__all__ = ['accumulate', 'accumulate_square', 'accumulate_mean']


def accumulate(variable: Tensor,
               update: Tensor,
               factor: Optional[Union[float, Tensor]] = None) -> Tensor:
    """
    Updates a Tensor inplace using `X = X + (b * Y)`.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    #variable and #update must be the same shape, but can be different types.

    Args:
        variable: Tensor
            Tensor to be updated.
        update: Tensor
            Value to update the variable
        factor: Optional[Union[float, Tensor]]
            Optional scalar to apply to update before the addition.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, variable, update)

    ins = {0: variable.id, 1: update.id}

    ov: _ir.OptimizerValue
    if isinstance(factor, Tensor):
        check_in_graph(g, factor)
        ins[2] = factor.id
        ov = _ir.OptimizerValue(0.0, False)
    elif isinstance(factor, float):
        ov = _ir.OptimizerValue(factor, True)
    else:
        ov = _ir.OptimizerValue()

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        ins,
        {
            0: g._create_tensor_id('Updated__' + variable.name),
        },
        _ir.AccumulationType.DampenedAdd
        if factor is not None else _ir.AccumulationType.Add,
        ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def accumulate_square(variable: Tensor,
                      update: Tensor,
                      factor: Optional[Union[float, Tensor]] = None) -> Tensor:
    """
    Updates a Tensor inplace using `X = X + (b * Y^2)`.

    Does not apply numpy broadcasting.
    Uses mixed precision poplibs operations.
    #variable and #update must be the same shape, but can be different types.

    Args:
        variable: Tensor
            Tensor to be updated.
        update: Tensor
            Value to update the variable
        factor: Optional[Union[float, Tensor]]
            Optional scalar to apply to update before the addition.
    Returns:
        updated: Tensor
            An alias to the variable.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, variable, update)

    ins = {0: variable.id, 1: update.id}

    ov: _ir.OptimizerValue
    if isinstance(factor, Tensor):
        check_in_graph(g, factor)
        ins[2] = factor.id
        ov = _ir.OptimizerValue(0.0, False)
    elif isinstance(factor, float):
        ov = _ir.OptimizerValue(factor, True)
    else:
        ov = _ir.OptimizerValue(1.0, True)

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        ins,
        {
            0: g._create_tensor_id('Updated__' + variable.name),
        },
        _ir.AccumulationType.DampenedAddSquare,
        ov,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def accumulate_mean(variable: Tensor, update: Tensor,
                    step: Union[float, Tensor]) -> Tensor:
    """
    Updates a Tensor inplace using `X = (step/(step+1)) * X + (1/(step+1)) * Y`.
    Intended to be used to keep track of the mean of a series of values. For example:
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
    #variable and #update must be the same shape, but can be different types.

    Args:
        variable: Tensor
            Tensor to be updated.
        update: Tensor
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

    step = variable._ensure_tensor(step, dtype=dtypes.float32)

    check_in_graph(g, variable, update, step)

    settings = ctx._get_op_settings('accumulate')
    op = pb_g.createConnectedOp_AccumulateOp(
        {
            0: variable.id,
            1: update.id,
            2: step.id
        },
        {
            0: g._create_tensor_id('Updated__' + variable.name),
        },
        _ir.AccumulationType.Mean,
        _ir.OptimizerValue(),
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
