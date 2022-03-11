# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor, Union
from popxl.remote_buffer import RemoteBuffer
from popxl import constant
from popxl.dtypes import uint32
from .utils import check_in_graph


@op_debug_context
def remote_store(remote_buffer: RemoteBuffer, offset: Union[int, Tensor],
                 t: Tensor) -> None:
    """
    Store a tensor in Streaming Memory.

    This operation stores the input tensor in the remote buffer residing in the off-chip Streaming Memory.

    This op is typically used to store different identically
    shaped tensors to the same remote buffer by specifying the offset.

    Op instances with matching ``remote_buffer_id`` (specified in ``remote_buffer``)
    will outline together, meaning that if different tensors are to be stored
    under the same remote buffer ID, a different ``offset`` value has to be supplied for
    each tensor.

    ``remote_buffer`` handles the relationship between ``remote_buffer_id``, ``shape``
    and ``dtype`` as ``shape`` and ``dtype`` needs to be fixed for each ``remote_buffer_id``.

    Each value of ``offset`` must be >= 0.

    If ``t`` is of rank ``x``, the remote buffer with ``remote_buffer_id`` will be of
    rank ``x+1``, where the new dimension (the row) will be of size ``entries``.

    Note:
        There is no data dependency (in the graph) between remote store and remote load.
        Thus, the remote load operator may end up before the remote store operator in the
        serialized graph.
        One way to circumvent this is by using ``with popxl.in_sequence(True)``

    See also:
        ``remote_buffer``, ``remote_load``, ``remote_load_``.

    Args:
        remote_buffer (RemoteBuffer): The handle to the remote buffer
        offset (Union[int, Tensor]): Integer or rank-0 tensor indicating what row/entry in the
          remote buffer to store to
        t (Tensor): Tensor to copy and store in the remote buffer
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    # Create tensors
    if isinstance(offset, int):
        offset = constant(offset, uint32, name="offset")

    check_in_graph(g, t=t, offset=offset)

    remote_buffer.validate_tensor_matches_buffer(t)

    settings = ctx._get_op_settings('remote_store')
    opid = _ir.OperatorIdentifier("ai.graphcore", "RemoteStore", 1,
                                  _ir.NumInputs(1, 2), 0)

    _ = pb_g.createConnectedOp_RemoteStoreOp({
        0: t.id,
        1: offset.id
    }, {}, opid, settings, remote_buffer.remote_buffer_id)
