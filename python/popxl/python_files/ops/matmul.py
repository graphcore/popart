# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional

import popart._internal.ir as _ir
from popxl import dtypes
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor

from .utils import check_in_graph, check_tensor_ipu_and_tile_set, convert_optional_float


@op_debug_context
def matmul(
    lhs: Tensor,
    rhs: Tensor,
    available_memory_proportion: Optional[float] = None,
    output_type: Optional[dtypes.dtype] = None,
    partials_type: Optional[dtypes.dtype] = None,
) -> Tensor:
    """Perform matrix multiplication of two tensors.

    Follows NumPy matrix multiplication rules for N-D tensors, see
    :py:func:`numpy.matmul`.

    Arguments must have the same dtype. Shapes must be compatible as defined by the
    NumPy matrix multiplication rules.

    See also `PyTorch Tensor.matmul <https://pytorch.org/docs/stable/generated/torch.Tensor.matmul.html>`__, `NumPy matmul <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`__, `ONNX MatMul <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul>`__.

    Args:
        lhs (Tensor): Left hand side of matrix multiplication.
        rhs (Tensor): Right hand side of matrix multiplication.
        available_memory_proportion (Optional[float]):
            The maximum proportion of available memory on each tile that this layer
            should consume temporarily during the course of the operation. Defaults to 1.0.
        output_type (Optional[dtypes.dtype], optional): 3
            Output datatype to enforce. Defaults to the dtype of lhs/rhs.
        partials_type (dtypes.dtype, optional): The type to use for partial results
             (float16, float32). Defaults to dtypes.float32.

    Returns:
        Tensor: The matrix product of `lhs` and `rhs`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    settings = ctx._get_op_settings("matmul")
    opid = _ir.OperatorIdentifier("ai.onnx", "MatMul", 9, _ir.NumInputs(2, 2), 1)

    if partials_type is None:
        partials_type = _convert_partials_type_from_str(
            g.ir._pb_ir.getSessionOptions().partialsTypeMatMuls
        )
    else:
        partials_type = _convert_partials_type_from_dtype(partials_type)

    # These two args can be none, in which case we want to send a no-opt optional.
    out_dtype = (
        _ir.OptionalDataType(output_type._pb_dtype)
        if output_type
        else _ir.OptionalDataType()
    )
    optional_memory_proportion = convert_optional_float(available_memory_proportion)

    op = pb_g.createConnectedOp_MatMulOp(
        {0: lhs.id, 1: rhs.id},
        {
            0: g._create_tensor_id("matmul_out"),
        },
        opid,
        settings,
        optional_memory_proportion,
        # pass empty serialisation as this is only required for SerializeMatMuls transform
        # which is not exposed in PopXL
        _ir.op.SerialiseSettings(),
        out_dtype,
        partials_type,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def matmul_pow2scaled(
    lhs: Tensor,
    rhs: Tensor,
    log2_scale: Tensor,
    available_memory_proportion: Optional[float] = None,
) -> Tensor:
    """Perform a scaled matrix multiplication between two tensors.

    Compute a matrix multiplication between `lhs` and `rhs`,
    then multiply the result by `pow2(log2_scale)`.

    The matrix multiply arguments must have either `popxl.float8_143`
    or `popxl.float8_152` dtype.
    The `log2_scale` argument must be of type `popxl.int8` and be in
    the range in [-32, 32).

    Follows NumPy matrix multiplication rules for N-D tensors, see
    :py:func:`numpy.matmul`.

    Args:
        lhs (Tensor): Left hand side of matrix multiplication.
        rhs (Tensor): Right hand side of matrix multiplication.
        log2_scale (Tensor): integer power-of-two exponent, where the matrix
            multiplication output is multiplied by pow2(log2_scale).
        available_memory_proportion (Optional[float]):
            The maximum proportion of available memory on each tile that this layer
            should consume temporarily during the course of the operation. Defaults to 1.0.
    Raises:
        TypeError: If the matrix multiply operand tensors do not have a dtype in
            `{popxl.float8_143, popxl.float8_152}`, or if the `log2_scale` tensor
            does not have dtype `popxl.int32`
        ValueError: If `log2_scale` is not a scalar tensor.
    """

    invalid_dtype = (
        lhs.dtype
        not in {
            dtypes.float8_143,
            dtypes.float8_152,
        }
    ) or (rhs.dtype not in {dtypes.float8_143, dtypes.float8_152})
    if invalid_dtype:
        raise TypeError(
            "Scaled matmul operands must be of dtype popxl.float8_143 or popxl.float8_152"
        )

    if log2_scale.rank > 0:
        raise ValueError("Log2 scale argument must be a scalar tensor")

    if log2_scale.dtype != dtypes.int32:
        raise TypeError("Log2 scale tensor must be of type popxl.int32")

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    settings = ctx._get_op_settings("matmul")
    opid = _ir.OperatorIdentifier("ai.onnx", "MatMul", 9, _ir.NumInputs(3, 3), 1)

    check_in_graph(g, lhs=lhs, rhs=rhs, log2_scale=log2_scale)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    # This arg can be none, in which case we want to send a no-opt optional.
    optional_memory_proportion = convert_optional_float(available_memory_proportion)
    op = pb_g.createConnectedOp_MatMulOp(
        {0: lhs.id, 1: rhs.id, 2: log2_scale.id},
        {
            0: g._create_tensor_id("matmul_pow2scaled_out"),
        },
        opid,
        settings,
        optional_memory_proportion,
        # pass empty serialisation as this is only required for SerializeMatMuls transform
        # which is not exposed in PopXL
        _ir.op.SerialiseSettings(),
        # For FP8 matmuls, output type and partials type must be set to half.
        _ir.OptionalDataType(_ir.DataType.FLOAT16),
        _ir.op.MatMulPartialsType.HALF,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def _convert_partials_type_from_dtype(type_: dtypes.dtype) -> _ir.op.MatMulPartialsType:
    """Convert the dtype to an _internal.ir partials type.

    Args:
        type_ (dtypes.dtype): The top level dtype

    Raises:
        ValueError: If not float[16|32]

    Returns:
        _ir.op.MatMulPartialsType: The internal partials type enum.
    """
    if type_ == dtypes.float16:
        return _ir.op.MatMulPartialsType.HALF
    elif type_ == dtypes.float32:
        return _ir.op.MatMulPartialsType.FLOAT
    else:
        raise ValueError(f"dtype {type_} is not valid, must be float16 or float32")


def _convert_partials_type_from_str(partials_type: str) -> _ir.op.MatMulPartialsType:
    """Convert SessionOption partialsTypeMatMuls to an _internal.ir partials type.

        Empty string defaults to `FLOAT`.

    Args:
        type_ (dtypes.dtype): The top level dtype

    Raises:
        ValueError: If not empty, 'float' or 'half'

    Returns:
        _ir.op.MatMulPartialsType: The internal partials type enum.
    """
    partials_type = partials_type.lower()
    if partials_type == "" or partials_type == "float":
        return _ir.op.MatMulPartialsType.FLOAT
    elif partials_type == "half":
        return _ir.op.MatMulPartialsType.HALF
    else:
        raise ValueError(f"str {partials_type} is not valid, must be 'float' or 'half'")
