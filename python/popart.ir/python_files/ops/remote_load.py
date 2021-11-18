# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from popart.ir.graph import Graph
from popart.ir.remote_buffer_handle import RemoteBufferHandle
from .utils import check_in_graph

__all__ = ["remote_load", "remote_load_"]


@op_debug_context
def remote_load(
        t: Tensor,
        offset: Optional[Tensor] = None,
        remote_buffer_handle: Optional[RemoteBufferHandle] = None) -> None:
    """Load a tensor from remote (off-chip) buffer.

    The tensor will be loaded from the memory location corresponding to
    `remote_buffer_id` (specified in the `remote_buffer_handle`),
    and will be stored in the memory location corresponding to `t`.

    The relationship between `offset` and `remote_buffer_id` is thoroughly
    described in `remote_store`.

    See also: `remote_buffer_handle`, `remote_store`, `remote_load_`

    Args:
        t (Tensor): This tensor will be cloned, and the loaded data will written to the clone.
        offset (Optional[Tensor], optional): Optional 0-rank Tensor.
          Specify the row in the remote buffer the inTensor will be loaded from.
          Defaults to None.
        remote_buffer_handle (Optional[RemoteBufferHandle], optional): The handle to the remote
          buffer. Defaults to None.
    Returns:
        Tensor: The tensor loaded from the remote buffer
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if offset is not None:
        check_in_graph(g, offset)

    remote_buffer_handle = prepare_remote_buffer(t, remote_buffer_handle, g)

    settings = ctx._get_op_settings('remote_load')
    opid = _ir.OperatorIdentifier("ai.graphcore", "RemoteLoad", 1,
                                  _ir.NumInputs(1, 2), 1)

    if offset is not None:
        op = pb_g.createConnectedOp_RemoteLoadOp(
            {
                0: t.id,
                1: offset.id
            }, {0: g._create_tensor_id("remote_load_out")}, opid, settings,
            remote_buffer_handle.remote_buffer_id)
    else:
        op = pb_g.createConnectedOp_RemoteLoadOp(
            {
                0: t.id,
            }, {0: g._create_tensor_id("remote_load_out")}, opid, settings,
            remote_buffer_handle.remote_buffer_id)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def remote_load_(
        t: Tensor,
        offset: Optional[Tensor] = None,
        remote_buffer_handle: Optional[RemoteBufferHandle] = None) -> None:
    """Load a tensor from remote (off-chip) buffer inplace.

    This op is identical to `remote_load` with the exception of how `t` is handled.
    In `remote_load` `t` is cloned and the output is written to the clone, whereas
    in this version `t` is written to directly.

    See also: `remote_buffer_handle`, `remote_store`, `remote_load`

    Args:
        t (Tensor): The tensor the loaded data will written to the clone.
        offset (Optional[Tensor], optional): Optional 0-rank Tensor.
          Specify the row in the remote buffer the inTensor will be loaded from.
          Defaults to None.
        remote_buffer_handle (Optional[RemoteBufferHandle], optional): The handle to the remote
          buffer. Defaults to None.
    Returns:
        Tensor: The tensor loaded from the remote buffer
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if offset is not None:
        check_in_graph(g, offset)

    remote_buffer_handle = prepare_remote_buffer(t, remote_buffer_handle, g)

    settings = ctx._get_op_settings('remote_load_inplace')
    opid = _ir.OperatorIdentifier("ai.graphcore", "RemoteLoadInplace", 1,
                                  _ir.NumInputs(1, 2), 1)

    if offset is not None:
        op = pb_g.createConnectedOp_RemoteLoadInplaceOp(
            {
                0: t.id,
                1: offset.id
            }, {0: g._create_tensor_id("remote_load_inplace_out")}, opid,
            settings, remote_buffer_handle.remote_buffer_id)
    else:
        op = pb_g.createConnectedOp_RemoteLoadInplaceOp(
            {
                0: t.id,
            }, {0: g._create_tensor_id("remote_load_inplace_out")}, opid,
            settings, remote_buffer_handle.remote_buffer_id)

    return Tensor._from_pb_tensor(op.outTensor(0))


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
        remote_buffer_handle = RemoteBufferHandle(
            remote_buffer_id=None,
            tensor_shape=t._pb_tensor.shape,
            tensor_dtype=t._pb_tensor.dtype,
            repeats=1)

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
