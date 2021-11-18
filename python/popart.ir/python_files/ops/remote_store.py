# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from popart.ir.remote_buffer_handle import RemoteBufferHandle
from .utils import check_in_graph

__all__ = ["remote_store"]


@op_debug_context
def remote_store(
        t: Tensor,
        offset: Optional[Tensor] = None,
        remote_buffer_handle: Optional[RemoteBufferHandle] = None) -> None:
    """Store the input tensor to a remote (off-chip) buffer.

    This Op is typically used when the user wants to store several different identically
    shaped tensors to the same remote buffer by specifying the offset (see below).

    Op instances with matching `remote_buffer_id` (specified in the `remote_buffer_handle`)
    will outline together, meaning that if multiple different tensors are to be stored
    under the same remote buffer ID, a different `offset` value has to be supplied for
    each tensor.

    The `remote_buffer_handle` handles the relationship between `remote_buffer_id`, shape
    and datatype as shape and datatype needs to be fixed for each `remote_buffer_id`.

    All `offset`s and `remote_buffer_id`s need to be >= 0.

    If `t` is of rank `x`, the remote buffer of a certain `remote_buffer_id` will be of
    rank `x+1`, where the new dimension (the row) will be of size `N`.

    See also: `remote_buffer_handle`, `remote_load`.

    Args:
        t (Tensor): Tensor to copy and store in the remote buffer.
        offset (Optional[Tensor], optional): Optional 0-rank Tensor.
          Specify the row in the remote buffer the inTensor will be written to.
          Defaults to None.
        remote_buffer_handle (Optional[RemoteBufferHandle], optional): The handle to the remote
          buffer. Defaults to None.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if offset is not None:
        check_in_graph(g, offset)

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

    settings = ctx._get_op_settings('remote_store')
    opid = _ir.OperatorIdentifier("ai.graphcore", "RemoteStore", 1,
                                  _ir.NumInputs(1, 2), 0)

    if offset is not None:
        _ = pb_g.createConnectedOp_RemoteStoreOp({
            0: t.id,
            1: offset.id
        }, {}, opid, settings, remote_buffer_handle.remote_buffer_id)
    else:
        _ = pb_g.createConnectedOp_RemoteStoreOp({
            0: t.id,
        }, {}, opid, settings, remote_buffer_handle.remote_buffer_id)
