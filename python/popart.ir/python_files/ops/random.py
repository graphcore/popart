# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from typing import Tuple
from popart.ir import dtypes
from .utils import check_in_graph, convert_optional_dtype

__all__ = ['random_uniform', 'random_normal']


def random_uniform(seed_tensor: Tensor,
                   shape: Tuple[int, ...],
                   low: float = 0.0,
                   high: float = 1.0,
                   dtype: dtypes.dtype = dtypes.float32):
    """
    Randomly sample from a uniform distribution with minimum value `low` and maximum value `high`.

    Note: not compatible with `IPUModel`.

    Args:
        seed_tensor (Tensor):
            Used to seed the probability distribution. Must have data type uint32 and shape (2,).
        shape (Tuple[int, ...]):
            Shape of output tensor
        low (float, optional): 
            Minimum value. Defaults to 0.0.
        high (float, optional): 
            Maximum value. Defaults to 1.0.
        dtype (dtypes.dtype, optional): 
            Data type of output tensor. Defaults to dtypes.float32.

    Returns:
        Tensor: tensor with elements sampled from a uniform distribution.
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, seed_tensor)

    settings = ctx._get_op_settings('random_uniform')
    opid = _ir.OperatorIdentifier("ai.onnx", "RandomUniform", 1,
                                  _ir.NumInputs(1, 1), 1)
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


def random_normal(seed_tensor: Tensor,
                  shape: Tuple[int, ...],
                  mean: float = 0.0,
                  std: float = 1.0,
                  dtype: dtypes.dtype = dtypes.float32):
    """
    Randomly sample from a normal distribution with `mean` and standard deviation `std`.

    Note: not compatible with `IPUModel`.

    Args:
        seed_tensor (Tensor):
            Used to seed the probability distribution. Must have data type uint32 and shape (2,).
        shape (Tuple[int, ...]):
            Shape of output tensor
        mean (float, optional): 
            Mean of distribution. Defaults to 0.0.
        std (float, optional): 
            Standard deviation of distribution. Defaults to 1.0.
        dtype (dtypes.dtype, optional): 
            Data type of output tensor. Defaults to dtypes.float32.

    Returns:
        Tensor: tensor with elements sampled from a normal distribution.
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, seed_tensor)

    settings = ctx._get_op_settings('random_normal')
    opid = _ir.OperatorIdentifier("ai.onnx", "RandomNormal", 1,
                                  _ir.NumInputs(1, 1), 1)
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
