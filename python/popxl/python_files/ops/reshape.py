# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Tuple
import numpy as np
import popart._internal.ir as _ir
from popxl.context import get_current_context, debug_context_frame_offset, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


def reshape_handle_negative_axis(t: Tensor,
                                 shape: Tuple[int, ...]) -> Tuple[int, ...]:
    replacement = t.nelms // np.prod(np.abs(shape))
    return tuple(axis if axis > 0 else replacement for axis in shape)


@op_debug_context
def reshape(t: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Reshape a tensor.

    See also `PyTorch Tensor.reshape <https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html>`__, `NumPy reshape <https://numpy.org/doc/stable/reference/generated/numpy.reshape.html>`__, `ONNX Reshape <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape>`__.

    Args:
        t (Tensor): The tensor to be reshaped.
        shape (Tuple[int, ...]): Tuple containing the shape of the output.

    Raises:
        ValueError: A ValueError will be raised if:

          - An invalid value is encountered in the shape.
          - If more than -1 is given in shape.

    Returns:
        Tensor: The reshaped tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if any(map(lambda axis: axis < -1, shape)) or shape.count(0) > 0:
        raise ValueError(
            f"Invalid shape value in reshape. Must be '>0' or '-1'. Provided {shape}."
        )

    if shape.count(-1) > 1:
        raise ValueError(
            f"Reshape shape can contain at most one '-1' value. Provided {shape}."
        )

    settings = ctx._get_op_settings('reshape')
    opid = _ir.OperatorIdentifier("ai.onnx", "Reshape", 5, _ir.NumInputs(1, 1),
                                  1)
    op = pb_g.createConnectedOp_ReshapeOp(
        {0: t.id}, {0: g._create_tensor_id(f"{t.name}_reshaped")}, opid,
        reshape_handle_negative_axis(t, shape), settings, False)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def reshape_(t: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Reshape a tensor (in-place).

    This is the in-place version of :func:`~popxl.ops.reshape`.

    Args:
        t (Tensor): The tensor to be reshaped.
        shape (Tuple[int, ...]): Tuple containing the shape of the output.

    Raises:
        ValueError: A ValueError will be raised if:

          - An invalid value is encountered in the shape.
          - If more than -1 is given in shape.

    Returns:
        Tensor: An alias of the input tensor, reshaped.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if any(map(lambda axis: axis < -1, shape)) or shape.count(0) > 0:
        raise ValueError(
            f"Invalid shape value in reshape. Must be '>0' or '-1'. Provided {shape}."
        )

    if shape.count(-1) > 1:
        raise ValueError(
            f"Reshape shape can contain at most one '-1' value. Provided {shape}."
        )

    settings = ctx._get_op_settings('reshape_inplace')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ReshapeInplace", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ReshapeInplaceOp(
        {0: t.id}, {0: g._create_tensor_id(f"{t.name}_reshaped")}, opid,
        reshape_handle_negative_axis(t, shape), settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


@debug_context_frame_offset(1)
def flatten(t: Tensor) -> Tensor:
    """
    Flatten a tensor.

    Internally this uses :func:`~popxl.ops.reshape`.

    See also `PyTorch Tensor.flatten <https://pytorch.org/docs/stable/generated/torch.Tensor.flatten.html>`__, `ONNX Flatten <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten>`__.

    Args:
        t (Tensor): The tensor to be flattened.

    Returns:
        Tensor: Tensor with 1-D shape.
    """
    return reshape(t, (-1, ))


@debug_context_frame_offset(1)
def flatten_(t: Tensor) -> Tensor:
    """
    Flatten a tensor in place.

    Internally this uses :func:`~popxl.ops.reshape_`.

    This is the in-place version of :func:`~popxl.ops.flatten`.

    Args:
        t (Tensor): The tensor to be flattened.

    Returns:
        Tensor: An alias of the input tensor with a 1-D shape.
    """
    return reshape_(t, (-1, ))
