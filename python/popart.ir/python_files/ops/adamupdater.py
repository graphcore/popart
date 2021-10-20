# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Union
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor

from .utils import check_in_graph

__all__ = ['adamupdater', 'lambupdater', 'adamaxupdater']


def addOptimizerValue(var, g, ins, index):
    if isinstance(var, Tensor):
        check_in_graph(g, var)
        ins[index] = var.id
        ov = _ir.OptimizerValue(0.0, False)
    elif isinstance(var, float):
        ov = _ir.OptimizerValue(var, True)
    else:
        ov = _ir.OptimizerValue()
    return ov


def create_adamupdater(acc_first_order: Tensor,
                       acc_second_order: Tensor,
                       ins,
                       mode,
                       weight: Optional[Tensor] = None,
                       time_step: Optional[Tensor] = None,
                       weight_decay: Optional[Union[float, Tensor]] = None,
                       beta1: Optional[Union[float, Tensor]] = None,
                       beta2: Optional[Union[float, Tensor]] = None,
                       epsilon: Union[float, Tensor] = 1e-07) -> Tensor:

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, acc_first_order, acc_second_order)
    if weight is not None:
        check_in_graph(g, weight)
    if time_step is not None:
        check_in_graph(g, time_step)
    if weight_decay is not None:
        check_in_graph(g, weight)

    outs = {
        0: g._create_tensor_id('Updater'),
    }
    wd = addOptimizerValue(weight_decay, g, ins, 4)
    b1 = addOptimizerValue(beta1, g, ins, 5)
    b2 = addOptimizerValue(beta2, g, ins, 6)
    eps = addOptimizerValue(epsilon, g, ins, 7)

    settings = ctx._get_op_settings('adamupdater')
    op = pb_g.createConnectedOp_AdamUpdaterOp(ins, outs, mode, wd, b1, b2, eps,
                                              settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


def adamupdater(acc_first_order: Tensor,
                acc_second_order: Tensor,
                weight: Optional[Tensor] = None,
                time_step: Optional[Tensor] = None,
                weight_decay: Optional[Union[float, Tensor]] = None,
                beta1: Optional[Union[float, Tensor]] = None,
                beta2: Optional[Union[float, Tensor]] = None,
                epsilon: Union[float, Tensor] = 1e-07) -> Tensor:
    """
    Calculate a updater term x for Adam as follows.
    accumulated bias corrected first order momentum (FP16/FP32) mc:
    mc = m / (1 - b1 ** t)  (without correction: mc = m)

    accumulated bias corrected second order momentum (FP16/FP32) vc:
    vc = v / (1 - b2 ** t)  (without correction: vc = v)

    updater term (FP16/FP32, with weight decay mode: decay and wd > 0.0) x:
    x = mc / (sqrt(vc) + eps) + wd * w

    updater term (FP16/FP32, without weight decay mode: decay) x:
    x = mc / (sqrt(vc) + eps)

    Args:
        acc_first_order: Tensor (m)
            First order momentum (FP16/FP32).
        acc_second_order: Tensor (v)
            Second order momentum (FP16/FP32).
        weight: Optional[Tensor] (w)
            Weight. Only required for weight_decay.
        time_step: Tensor (t)
            Time step. Providing this tensor enables bias correction.
        weight_decay: Optional[Union[float, Tensor]] = None
            Optional scalar to apply weight decay.
        beta1: Optional[Union[float, Tensor]] = None
            Only required in bias correction for m
        beta2: Optional[Union[float, Tensor]] = None
            Only required in bias correction for v
        epsilon: Union[float, Tensor] = 1e-07
            Scalar to calculate updater.

    Returns:
        updater: Tensor
            An updater to update weight.
    """
    ins = {1: acc_first_order.id, 2: acc_second_order.id}
    if weight_decay is not None and weight is None:
        raise ValueError("Weight decay requires weight to be not None.")
    if weight is not None and weight_decay is not None:
        ins[0] = weight.id

    if time_step is not None and (beta1 is None or beta2 is None):
        raise ValueError(
            "Bias correction requires both beta1 and beta2 not None.")
    if time_step is not None and beta1 is not None and beta2 is not None:
        ins[3] = time_step.id
        adam_mode = _ir.AdamMode.Adam
    else:
        adam_mode = _ir.AdamMode.AdamNoBias

    return create_adamupdater(acc_first_order, acc_second_order, ins,
                              adam_mode, weight, time_step, weight_decay,
                              beta1, beta2, epsilon)


def lambupdater(acc_first_order: Tensor,
                acc_second_order: Tensor,
                weight: Optional[Tensor] = None,
                time_step: Optional[Tensor] = None,
                weight_decay: Optional[Union[float, Tensor]] = None,
                beta1: Optional[Union[float, Tensor]] = None,
                beta2: Optional[Union[float, Tensor]] = None,
                epsilon: Union[float, Tensor] = 1e-07) -> Tensor:
    """
    Calculate a updater term x for Lamb as follows.
    accumulated bias corrected first order momentum (FP16/FP32) mc:
    mc = m / (1 - b1 ** t)  (without correction: mc = m)

    accumulated bias corrected second order momentum (FP16/FP32) vc:
    vc = v / (1 - b2 ** t)  (without correction: vc = v)

    updater term (FP16/FP32, with weight decay mode: decay and wd > 0.0) x:
    x = mc / (sqrt(vc) + eps) + wd * w

    updater term (FP16/FP32, without weight decay mode: decay) x:
    x = mc / (sqrt(vc) + eps)

    Args:
        acc_first_order: Tensor (m)
            First order momentum (FP16/FP32).
        acc_second_order: Tensor (v)
            Second order momentum (FP16/FP32).
        weight: Optional[Tensor] (w)
            Weight. Only required for weight_decay.
        time_step: Tensor (t)
            Time step. Providing this tensor enables bias correction.
        weight_decay: Optional[Union[float, Tensor]] = None
            Optional scalar to apply weight decay.
        beta1: Optional[Union[float, Tensor]] = None
            Only required in bias correction for m.
        beta2: Optional[Union[float, Tensor]] = None
            Only required in bias correction for v.
        epsilon: Union[float, Tensor] = 1e-07
            Scalar to calculate updater.
    Returns:
        updater: Tensor
            An updater to update weight.
    """
    ins = {1: acc_first_order.id, 2: acc_second_order.id}
    if weight_decay is not None and weight is None:
        raise ValueError("Weight decay requires weight to be not None.")

    if weight is not None and weight_decay is not None:
        ins[0] = weight.id

    if time_step is not None and (beta1 is None or beta2 is None):
        raise ValueError(
            "Bias correction requires both beta1 and beta2 not None.")
    if time_step is not None and beta1 is not None and beta2 is not None:
        ins[3] = time_step.id
        adam_mode = _ir.AdamMode.Lamb
    else:
        adam_mode = _ir.AdamMode.LambNoBias

    return create_adamupdater(acc_first_order, acc_second_order, ins,
                              adam_mode, weight, time_step, weight_decay,
                              beta1, beta2, epsilon)


def adamaxupdater(acc_first_order: Tensor,
                  acc_second_order: Tensor,
                  weight: Optional[Tensor] = None,
                  time_step: Tensor = None,
                  weight_decay: Optional[Union[float, Tensor]] = None,
                  beta1: Union[float, Tensor] = 0.9,
                  epsilon: Union[float, Tensor] = 1e-07) -> Tensor:
    """
    Calculate a updater term x for Adamax as follows.
    accumulated bias corrected first order momentum (FP16/FP32) mc:
    mc = m / (1 - b1 ** t)
    updater term (FP16/FP32, with weight decay mode: decay and wd > 0.0) x:
    x = mc / (vc + eps) + wd * w

    updater term (FP16/FP32, without weight decay mode: decay) x:
    x = mc / (vc + eps)


    Args:
        acc_first_order: Tensor (m)
            First order momentum (FP16/FP32).
        acc_second_order: Tensor (v)
            Second order momentum (FP16/FP32).
        weight: Optional[Tensor] (w)
            Weight. Only required for weight_decay.
        time_step: Tensor (t)
            Time step.
        weight_decay: Optional[Union[float, Tensor]] = None
            Optional scalar to apply weight decay
        beta1: Union[float, Tensor] = 0.9
            Scalar to do bias correction for m
        epsilon: Union[float, Tensor] = 1e-07
            Scalar to calculate updater.

    Returns:
        updater: Tensor
            An updater to update weight.
    """
    ins = {1: acc_first_order.id, 2: acc_second_order.id}
    if weight_decay is not None and weight is None:
        raise ValueError("Weight decay requires weight to be not None.")
    if weight is not None and weight_decay is not None:
        ins[0] = weight.id
    if time_step is None:
        raise ValueError("AdaMax requires time_step not None.")
    else:
        ins[3] = time_step.id
        adam_mode = _ir.AdamMode.AdaMax

    return create_adamupdater(acc_first_order, acc_second_order, ins,
                              adam_mode, weight, time_step, weight_decay,
                              beta1, None, epsilon)
