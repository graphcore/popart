# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional

import popart._internal.ir as _ir
from popart.ir.tensor import Tensor
from popart.ir.graph import Graph
from popart.ir.dtypes import dtype
from popart.ir.remote_buffer_handle import RemoteBufferHandle


def cast_if_needed(t: Tensor, data_type: dtype) -> Tensor:
    from popart.ir.ops.cast import cast
    if t.dtype != data_type:
        return cast(t, data_type)
    return t


def check_in_graph(graph: Graph, *tensors: Tensor):
    """Checks if tensors are in graph. If not, raises a ValueError."""
    for tensor in tensors:
        if tensor not in graph:
            raise ValueError(
                f"{tensor} is not in the current Graph {graph.name}.")


def handle_negative_axis(t: Tensor, axis: int) -> int:
    return len(t.shape) + axis if axis < 0 else axis


def convert_optional_float(v: Optional[float]):
    return _ir.OptionalFloat(v) if v is not None else _ir.OptionalFloat()


def convert_optional_int(v: Optional[int]):
    return _ir.OptionalInt(v) if v is not None else _ir.OptionalInt()


def convert_optional_dtype(dt: Optional[dtype]):
    return _ir.OptionalDataType(
        dt._pb_dtype) if dt is not None else _ir.OptionalDataType()


def prepare_remote_buffer(t: Tensor,
                          remote_buffer_handle: Optional[RemoteBufferHandle],
                          g: Graph) -> RemoteBufferHandle:
    """Prepare the remote buffer.

    Args:
        t (Tensor): Input tensor to the op.
        remote_buffer_handle (Optional[RemoteBufferHandle]): If set:
          The remote buffer handle to use in the preparation
        g (Graph): The graph to set the remote buffer info to

    Raises:
        ValueError: If there is a shape or type mismatch between `t` and
          `remote_buffer_handle`

    Returns:
        RemoteBufferHandle: The remote buffer handle used in the preparation.
    """
    if remote_buffer_handle is None:
        shape = t._pb_tensor.info.shape()
        d_type = dtype.as_dtype(t._pb_tensor.info.data_type_lcase())
        # Check for existing buffer handles
        existing_buffers = RemoteBufferHandle._buffers
        for _, rbh in existing_buffers.items():
            if rbh.tensor_shape == shape and rbh.tensor_dtype == d_type:
                remote_buffer_handle = rbh
                break

        if remote_buffer_handle is None:
            # Create handle if not found
            remote_buffer_handle = RemoteBufferHandle(remote_buffer_id=None,
                                                      tensor_shape=shape,
                                                      tensor_dtype=d_type,
                                                      repeats=1)

    # The remote buffer handle may be set, and may have empty shape and dtype
    if remote_buffer_handle.tensor_shape is None:
        remote_buffer_handle.tensor_shape = t._pb_tensor.info.shape()
    if remote_buffer_handle.tensor_dtype is None:
        remote_buffer_handle.tensor_dtype = dtype.as_dtype(
            t._pb_tensor.info.data_type_lcase())

    info = _ir.TensorInfo(remote_buffer_handle.tensor_dtype._pb_dtype,
                          remote_buffer_handle.tensor_shape)
    if t._pb_tensor.info.dataType() != info.dataType():
        raise ValueError(
            f"DataType of {t.id} ({t._pb_tensor.info.dataType()}) "
            f"does not match that of the RemoteBufferHandle ({info.dataType()})"
        )
    if t._pb_tensor.info.shape() != info.shape():
        raise ValueError(
            f"DataType of {t.id} ({t._pb_tensor.info.shape()}) "
            f"does not match that of the RemoteBufferHandle ({info.shape()})")

    g._ir._pb_ir.setRemoteBufferInfo(
        remote_buffer_handle.remote_buffer_id,
        _ir.RemoteBufferInfo(info, remote_buffer_handle.repeats))

    return remote_buffer_handle
