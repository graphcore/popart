# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.dtypes import dtype
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph
from popxl import float8_143, float8_152, float16, float32


@op_debug_context
def cast(t: Tensor, data_type: dtype) -> Tensor:
    """
    Cast a tensor to a specific data type.

    This operation casts tensor `t` to data type `data_type`.

    See also `ONNX Cast <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast>`__.

    Args:
        t (Tensor):
            The tensor to be cast.
        data_type (popxl.dtypes.dtype):
            The dtype to cast to.
    Returns:
        Tensor: The tensor cast to the specified type.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("cast")
    opid = _ir.OperatorIdentifier("ai.onnx", "Cast", 9, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_CastOp(
        {0: t.id},
        {0: g._create_tensor_id(f"{t.name}_{data_type._name}")},
        _to=data_type._pb_dtype,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def pow2scale_then_cast(t: Tensor, log2_scale: Tensor, data_type: dtype) -> Tensor:
    """Add a fused operation `cast(src * pow2(log2_scale), dtype)` to cast to floating point 8 data type.

    See the PopXL documentation on floating point 8 types for more details.

    Args:
        t (Tensor): Tensor to convert.
        log2_scale (Tensor): Scalar Tensor to use as the log 2 scale.
        data_type (dtype): Data type to convert to. Must be float8_143 or float8_152

    Raises:
        TypeError: If `data_type` is not of type float8_143 or float8_152.

    Returns:
        Tensor: The converted float 8 tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    if data_type not in [float8_143, float8_152]:
        raise TypeError(f"Data type {data_type} not supported for {__name__}")

    check_in_graph(g, t=t, log2_scale=log2_scale)

    settings = ctx._get_op_settings("pow2scale_then_cast")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "Pow2ScaleThenCast", 1, _ir.NumInputs(2, 2), 1
    )
    op = pb_g.createConnectedOp_Pow2ScaleThenCastOp(
        {0: t.id, 1: log2_scale.id},
        {0: g._create_tensor_id(f"{t.name}_{data_type._name}")},
        _to=data_type._pb_dtype,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def cast_then_pow2scale(t: Tensor, log2_scale: Tensor, data_type: dtype):
    """Add a fused operation `cast(X, dtype) * pow2(log2_scale)` to cast from floating point 8 type.

    See the PopXL documentation on floating point 8 types for more details.

    Args:
        t (Tensor): Tensor to convert.
        log2_scale (Tensor): Scalar Tensor to use as the log 2 scale.
        data_type (dtype): Data type to convert to. Must be float16 or float32.

    Raises:
        TypeError: If `data_type` is not of type float16 or float32.

    Returns:
        Tensor: The converted float16 or float32 tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    if data_type not in [float16, float32]:
        raise TypeError(f"Data type {data_type} not supported for {__name__}")

    check_in_graph(g, t=t, log2_scale=log2_scale)

    settings = ctx._get_op_settings("cast_then_pow2scale")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "CastThenPow2Scale", 1, _ir.NumInputs(2, 2), 1
    )
    op = pb_g.createConnectedOp_CastThenPow2ScaleOp(
        {0: t.id, 1: log2_scale.id},
        {0: g._create_tensor_id(f"{t.name}_{data_type._name}")},
        _to=data_type._pb_dtype,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
