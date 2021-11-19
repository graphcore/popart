# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from popart.ir.graph import Graph
from popart.ir.remote_buffer_handle import RemoteBufferHandle
from .utils import check_in_graph, prepare_remote_buffer

__all__ = ["remote_load", "remote_load_"]


@op_debug_context
def remote_load(
        t: Tensor,
        offset: Optional[Tensor] = None,
        remote_buffer_handle: Optional[RemoteBufferHandle] = None) -> Tensor:
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
        remote_buffer_handle: Optional[RemoteBufferHandle] = None) -> Tensor:
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
