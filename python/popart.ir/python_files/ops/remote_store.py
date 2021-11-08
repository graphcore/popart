# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ["remote_store"]


def remote_store(t: Tensor,
                 offset: Optional[Tensor] = None,
                 remote_buffer_id: int = -1) -> None:
    """
    Store the input tensor to a remote (off-chip) buffer.

    This Op is typically used when the user wants to store several different identically
    shaped tensors to the same remote buffer by specifying the offset (see below).

    Op instances with matching `remoteBufferId` will outline together, meaning that if
    multiple different tensors are to be stored under the same remote buffer ID, a
    different `offset` value has to be supplied for each tensor.

    All `offsets` and `RemoteBufferIds` need to be >= 0.

    If `t` is of rank `x`, the remote buffer of a certain `RemoteBufferId` will be of rank
    `x+1`, where the new dimension (the row) will be of size `N`.

    Args:
        t: Tensor
            Tensor to copy and store in the remote buffer.
        offset: Optional[Tensor]
            Optional 0-rank Tensor.
            Specify the row in the remote buffer the inTensor will be written to.
        remote_buffer_id: Optional[int]
            The buffer id to store the the t to.
    """
    if remote_buffer_id == -1:
        raise NotImplementedError(
            "remote_buffer_id = -1 (automatic RemoteSetup) not supported yet")

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if offset is not None:
        check_in_graph(g, offset)

    g._ir._pb_ir.setRemoteBufferInfo(
        remote_buffer_id, _ir.RemoteBufferInfo(t._pb_tensor.info, 1))

    settings = ctx._get_op_settings('remote_store')
    opid = _ir.OperatorIdentifier("ai.graphcore", "RemoteStore", 1,
                                  _ir.NumInputs(1, 2), 0)

    if offset is not None:
        _ = pb_g.createConnectedOp_RemoteStoreOp({
            0: t.id,
            1: offset.id
        }, {}, opid, settings, remote_buffer_id)
    else:
        _ = pb_g.createConnectedOp_RemoteStoreOp({
            0: t.id,
        }, {}, opid, settings, remote_buffer_id)
