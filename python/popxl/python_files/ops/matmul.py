# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
from enum import Enum

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from popxl import dtypes
from .utils import check_in_graph, convert_optional_float, check_tensor_ipu_and_tile_set


class SerialiseMode(Enum):
    """Enum for the serialise mode.

    Args:
        Enum (enum): The serialisation type.
    """
    NoSerialisation = "NoSerialisation"
    ReducingDim = "ReducingDim"
    InputChannels = "InputChannels"
    OutputChannels = "OutputChannels"


@op_debug_context
def matmul(lhs: Tensor,
           rhs: Tensor,
           available_memory_proportion: Optional[float] = None,
           serialise_mode: SerialiseMode = SerialiseMode.NoSerialisation,
           serialise_factor: int = 1,
           output_type: Optional[dtypes.dtype] = None,
           partials_type: Optional[dtypes.dtype] = None) -> Tensor:
    """Perform matrix multiplication of two tensors.

    Follows NumPy matrix multiplication rules for N-D tensors, see
    :py:func:`numpy.matmul`.

    Arguments must have the same dtype. Shapes must be compatible as defined by the
    NumPy matrix multiplication rules.

    This is similar to :onnxop:`MatMul`.

    Args:
        lhs (Tensor): Left hand side of matrix multiplication.
        rhs (Tensor): Right hand side of matrix multiplication.
        available_memory_proportion (Optional[float]):
            The maximum proportion of available memory on each tile that this layer
            should consume temporarily during the course of the operation. Defaults to 1.0.
        serialise_mode (SerialiseMode, optional):
            The serialisation mode to use (NoSerialisation, ReducingDim, InputChannels, OutputChannels).
            Defaults to SerialiseMode.NoSerialisation.
        serialise_factor (int, optional):
            The factor to serialise by. Defaults to 1.
        output_type (Optional[dtypes.dtype], optional): 3
            Output datatype to enforce. Defaults to the dtype of lhs/rhs.
        partials_type (dtypes.dtype, optional): The type to use for partial results
             (float16, float32). Defaults to dtypes.float32.

    Returns:
        Tensor: The matrix product of lhs and rhs.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    settings = ctx._get_op_settings('matmul')
    opid = _ir.OperatorIdentifier("ai.onnx", "MatMul", 9, _ir.NumInputs(2, 2),
                                  1)

    if partials_type is None:
        partials_type = g.ir._pb_ir.getSessionOptions().partialsTypeMatMuls
        if partials_type == "":
            partials_type = _ir.op.MatMulPartialsType.FLOAT
    else:
        partials_type = _convert_partials_type(partials_type)

    # These two args can be none, in which case we want to send a no-opt optional.
    out_dtype = _ir.OptionalDataType(
        output_type._pb_dtype) if output_type else _ir.OptionalDataType()
    optional_memory_proportion = convert_optional_float(
        available_memory_proportion)

    serialise_settings = _convert_serialisation_settings(
        serialise_mode, serialise_factor)
    op = pb_g.createConnectedOp_MatMulOp({
        0: lhs.id,
        1: rhs.id
    }, {
        0: g._create_tensor_id("matmul_out"),
    }, opid, settings, optional_memory_proportion, serialise_settings,
                                         out_dtype, partials_type)

    return Tensor._from_pb_tensor(op.outTensor(0))


def _convert_serialisation_settings(mode: SerialiseMode,
                                    factor: int) -> _ir.op.SerialiseSettings:
    """Convert the given serialise enum and factor to ir.op.SerialiseSettings.

    Args:
        mode (SerialiseMode): The mode to use
        factor (int): The factor to serialsise by.

    Returns:
        _ir.op.SerialiseSettings: An internal IR object representing
             the serialisation settings.
    """
    serialise_settings = _ir.op.SerialiseSettings()
    if mode == SerialiseMode.InputChannels:
        serialise_settings.mode = _ir.op.SerialiseSettingsMode.InputChannels
    elif mode == SerialiseMode.ReducingDim:
        serialise_settings.mode = _ir.op.SerialiseSettingsMode.ReducingDim
    elif mode == SerialiseMode.OutputChannels:
        serialise_settings.mode = _ir.op.SerialiseSettingsMode.OutputChannels
    else:
        serialise_settings.mode = _ir.op.SerialiseSettingsMode.NoSerialisation
    serialise_settings.factor = factor
    return serialise_settings


def _convert_partials_type(type_: dtypes.dtype) -> _ir.op.MatMulPartialsType:
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
        raise ValueError(
            f"dtype {type_} is not valid, must be float16 or float32")
