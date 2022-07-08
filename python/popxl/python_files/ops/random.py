# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from typing import Tuple
from popxl import dtypes
from .utils import check_in_graph, convert_optional_dtype


@op_debug_context
def random_uniform(
    seed_tensor: Tensor,
    shape: Tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: dtypes.dtype = dtypes.float32,
):
    """
    Randomly sample from a uniform distribution.

    This operation will sample uniformly from a range with minimum value `low`
    and maximum value `high`.

    Note: not compatible with IPU Model.

    Args:
        seed_tensor (Tensor):
            A tensor used to seed the probability distribution. Must have data type uint32 and shape (2,).
        shape (Tuple[int, ...]):
            The shape of the output tensor.
        low (float, optional):
            Minimum value. Defaults to 0.0.
        high (float, optional):
            Maximum value. Defaults to 1.0.
        dtype (dtypes.dtype, optional):
            Data type of output tensor. Defaults to dtypes.float32.

    Returns:
        Tensor: A new tensor with element values sampled from a uniform distribution.
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, seed_tensor=seed_tensor)

    settings = ctx._get_op_settings("random_uniform")
    opid = _ir.OperatorIdentifier("ai.onnx", "RandomUniform", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_RandomUniformOp(
        {0: seed_tensor.id},
        {0: g._create_tensor_id("random_uniform_out")},
        shape_=shape,
        low_=low,
        high_=high,
        dataType_=convert_optional_dtype(dtype),
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def random_normal(
    seed_tensor: Tensor,
    shape: Tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: dtypes.dtype = dtypes.float32,
):
    """
    Randomly sample from a normal distribution.

    The mean and standard deviation of the distribution is specified by `mean` and `std` respectively.

    Note: not compatible with IPU Model.

    Args:
        seed_tensor (Tensor):
            A tensor used to seed the probability distribution. Must have data type uint32 and shape (2,).
        shape (Tuple[int, ...]):
            The shape of the output tensor.
        mean (float, optional):
            Mean of the distribution. Defaults to 0.0.
        std (float, optional):
            Standard deviation of the distribution. Defaults to 1.0.
        dtype (dtypes.dtype, optional):
            Data type of output tensor. Defaults to dtypes.float32.

    Returns:
        Tensor: A new tensor with elements sampled from a normal distribution.
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, seed_tensor=seed_tensor)

    settings = ctx._get_op_settings("random_normal")
    opid = _ir.OperatorIdentifier("ai.onnx", "RandomNormal", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_RandomNormalOp(
        {0: seed_tensor.id},
        {0: g._create_tensor_id("random_normal_out")},
        shape_=shape,
        mean_=mean,
        scale_=std,
        dataType_=convert_optional_dtype(dtype),
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
