# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Union
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor

from ..utils import check_in_graph, check_tensor_ipu_and_tile_set
from .utils import handle_optimizer_value


def weight_decay_is_required(weight_decay: Union[None, float, Tensor]):
    if isinstance(weight_decay, Tensor):
        return True
    if weight_decay is not None:
        return weight_decay != 0.0
    return False


def create_adamupdater(
    acc_first_order: Tensor,
    acc_second_order: Tensor,
    ins,
    mode,
    weight: Optional[Tensor] = None,
    time_step: Optional[Tensor] = None,
    weight_decay: Optional[Union[float, Tensor]] = None,
    beta1: Optional[Union[float, Tensor]] = None,
    beta2: Optional[Union[float, Tensor]] = None,
    epsilon: Union[float, Tensor] = 1e-07,
) -> Tensor:

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(
        acc_first_order=acc_first_order, acc_second_order=acc_second_order
    )

    if isinstance(weight, Tensor):
        tensors_to_check["weight"] = weight
    if isinstance(time_step, Tensor):
        tensors_to_check["time_step"] = time_step
    if isinstance(weight_decay, Tensor):
        tensors_to_check["weight_decay"] = weight_decay

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    outs = {
        0: g._create_tensor_id("Updater"),
    }
    wd = handle_optimizer_value(weight_decay, ins, 4)
    b1 = handle_optimizer_value(beta1, ins, 5)
    b2 = handle_optimizer_value(beta2, ins, 6)
    eps = handle_optimizer_value(epsilon, ins, 7)

    settings = ctx._get_op_settings("adamupdater")
    op = pb_g.createConnectedOp_AdamUpdaterOp(
        ins, outs, mode, wd, b1, b2, eps, settings
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def adam_updater(
    acc_first_order: Tensor,
    acc_second_order: Tensor,
    weight: Optional[Tensor] = None,
    time_step: Optional[Tensor] = None,
    weight_decay: Optional[Union[float, Tensor]] = None,
    beta1: Optional[Union[float, Tensor]] = None,
    beta2: Optional[Union[float, Tensor]] = None,
    epsilon: Union[float, Tensor] = 1e-07,
) -> Tensor:
    """
    Calculate an updater term to update the weights for Adam.

    accumulated bias corrected first order momentum (FP16/FP32) mc:
    mc = m / (1 - b1 ** t)  (without correction: mc = m)

    accumulated bias corrected second order momentum (FP16/FP32) vc:
    vc = v / (1 - b2 ** t)  (without correction: vc = v)

    updater term (FP16/FP32, with weight decay mode: decay and wd > 0.0) x:
    x = mc / (sqrt(vc) + eps) + wd * w

    updater term (FP16/FP32, without weight decay mode: decay) x:
    x = mc / (sqrt(vc) + eps)

    Note: `time_step` will be incremented by 1.

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

    Raises:
        ValueError: A ValueError will be raised if:
            - weight_decay is set and weight is None.
            - time_step set None and beta1 and beta2 is not set
              (no bias correction can take place).

    Returns:
        updater: Tensor
            An updater to update weight.
    """
    ins = {1: acc_first_order.id, 2: acc_second_order.id}
    if weight_decay_is_required(weight_decay) and weight is None:
        raise ValueError("Weight decay requires weight to be not None.")
    if weight_decay_is_required(weight_decay) and weight is not None:
        ins[0] = weight.id

    if time_step is not None and (beta1 is None or beta2 is None):
        raise ValueError("Bias correction requires both beta1 and beta2 not None.")
    if time_step is not None and beta1 is not None and beta2 is not None:
        ins[3] = time_step.id
        adam_mode = _ir.AdamMode.Adam
    else:
        adam_mode = _ir.AdamMode.AdamNoBias

    return create_adamupdater(
        acc_first_order,
        acc_second_order,
        ins,
        adam_mode,
        weight,
        time_step,
        weight_decay,
        beta1,
        beta2,
        epsilon,
    )


@op_debug_context
def lamb_updater(
    acc_first_order: Tensor,
    acc_second_order: Tensor,
    weight: Optional[Tensor] = None,
    time_step: Optional[Tensor] = None,
    weight_decay: Optional[Union[float, Tensor]] = None,
    beta1: Optional[Union[float, Tensor]] = None,
    beta2: Optional[Union[float, Tensor]] = None,
    epsilon: Union[float, Tensor] = 1e-07,
) -> Tensor:
    """
    Calculate an updater term to update the weights for LAMB.

    accumulated bias corrected first order momentum (FP16/FP32) mc:
    mc = m / (1 - b1 ** t)  (without correction: mc = m)

    accumulated bias corrected second order momentum (FP16/FP32) vc:
    vc = v / (1 - b2 ** t)  (without correction: vc = v)

    updater term (FP16/FP32, with weight decay mode: decay and wd > 0.0) x:
    x = mc / (sqrt(vc) + eps) + wd * w

    updater term (FP16/FP32, without weight decay mode: decay) x:
    x = mc / (sqrt(vc) + eps)

    Note: `time_step` will be incremented by one.

    Args:
        acc_first_order (Tensor):
            First order momentum (FP16/FP32) (m).
        acc_second_order (Tensor):
            Second order momentum (FP16/FP32) (v).
        weight (Optional[Tensor], optional):
            Weight (v). Only required for weight_decay.
            Defaults to None.
        time_step (Optional[Tensor], optional):
            Time step (v). Providing this tensor enables bias correction.
            Defaults to None.
        weight_decay (Optional[Union[float, Tensor]], optional):
            Optional scalar to apply weight decay.
            Defaults to None.
        beta1 (Optional[Union[float, Tensor]], optional):
            Only required in bias correction for m.
            Defaults to None.
        beta2 (Optional[Union[float, Tensor]], optional):
            Only required in bias correction for v.
            Defaults to None.
        epsilon (Union[float, Tensor], optional):
            Scalar to calculate updater.
            Defaults to 1e-07.

    Raises:
        ValueError: A ValueError will be raised if:
            - weight_decay is set and weight is None.
            - time_step set None and beta1 and beta2 is not set
              (no bias correction can take place).

    Returns:
        Tensor:
            An updater to update weight.
    """
    ins = {1: acc_first_order.id, 2: acc_second_order.id}
    if weight_decay_is_required(weight_decay) and weight is None:
        raise ValueError("Weight decay requires weight to be not None.")
    if weight_decay_is_required(weight_decay) and weight is not None:
        ins[0] = weight.id

    if time_step is not None and (beta1 is None or beta2 is None):
        raise ValueError("Bias correction requires both beta1 and beta2 not None.")
    if time_step is not None and beta1 is not None and beta2 is not None:
        ins[3] = time_step.id
        adam_mode = _ir.AdamMode.Lamb
    else:
        adam_mode = _ir.AdamMode.LambNoBias

    return create_adamupdater(
        acc_first_order,
        acc_second_order,
        ins,
        adam_mode,
        weight,
        time_step,
        weight_decay,
        beta1,
        beta2,
        epsilon,
    )


@op_debug_context
def adamax_updater(
    acc_first_order: Tensor,
    acc_second_order: Tensor,
    weight: Optional[Tensor] = None,
    time_step: Tensor = None,
    weight_decay: Optional[Union[float, Tensor]] = None,
    beta1: Union[float, Tensor] = 0.9,
    epsilon: Union[float, Tensor] = 1e-07,
) -> Tensor:
    """
    Calculate an updater term to update the weights for Adamax.

    accumulated bias corrected first order momentum (FP16/FP32) mc:
    mc = m / (1 - b1 ** t)
    updater term (FP16/FP32, with weight decay mode: decay and wd > 0.0) x:
    x = mc / (vc + eps) + wd * w

    updater term (FP16/FP32, without weight decay mode: decay) x:
    x = mc / (vc + eps)

    Note: `time_step` will be incremented by one.

    Args:
        acc_first_order (Tensor):
            First order momentum (FP16/FP32) (m).
        acc_second_order (Tensor):
            Second order momentum (FP16/FP32) (v).
        weight (Optional[Tensor]):
            Weight (w). Only required for weight_decay.
        time_step (Tensor):
            Time step (t).
        weight_decay (Optional[Union[float, Tensor]]):
            Optional scalar to apply weight decay. Defaults to None
        beta1 (Union[float, Tensor]):
            Scalar to do bias correction for m. Defaults to 0.9
        epsilon (Union[float, Tensor]):
            Scalar to calculate updater. Defaults to 1e-07

    Raises:
        ValueError: A ValueError will be raised if:
            - weight_decay is set and weight is None.
            - time_step is None.

    Returns:
        updater: Tensor
            An updater to update weight.
    """
    ins = {1: acc_first_order.id, 2: acc_second_order.id}
    if weight_decay_is_required(weight_decay) and weight is None:
        raise ValueError("Weight decay requires weight to be not None.")

    if weight_decay_is_required(weight_decay) and weight is not None:
        ins[0] = weight.id

    if time_step is None:
        raise ValueError("AdaMax requires time_step not None.")
    else:
        ins[3] = time_step.id
        adam_mode = _ir.AdamMode.AdaMax

    return create_adamupdater(
        acc_first_order,
        acc_second_order,
        ins,
        adam_mode,
        weight,
        time_step,
        weight_decay,
        beta1,
        None,
        epsilon,
    )
