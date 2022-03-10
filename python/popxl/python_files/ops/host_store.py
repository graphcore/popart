# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context, gmg
from popxl.tensor import Tensor
from popxl.streams import DeviceToHostStream

from .utils import check_in_graph


@op_debug_context
def host_store(d2h_stream: DeviceToHostStream, t: Tensor) -> None:
    """
    Transfer a tensor from the IPU to the host.

    This operation represents the transfer of data from the IPU to the
    host. It uses the existing device to host transfers created when building
    the IR, but defers the actual poplar::Copy until the op itself runs. This
    allows the copy to be scheduled as part of the normal op scheduling.

    Data is received on the host via the :py:class:`IStepIO` object passed to
    `session.run()`.

    Args:
        t (Tensor): The input tensor to copy to host.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    check_in_graph(gmg(), d2h_stream=d2h_stream._stream_tensor)

    if d2h_stream.dtype != t.dtype:
        raise ValueError(
            f'dtype of stream {d2h_stream.tensor_id} `{d2h_stream.dtype}` does not match dtype of provided tensor `{t.dtype}`'
        )
    if d2h_stream.shape != t.shape:
        raise ValueError(
            f'shape of stream {d2h_stream.tensor_id} `{d2h_stream.shape}` does not match shape of provided tensor `{t.shape}`'
        )

    opid = _ir.OperatorIdentifier("ai.graphcore", "HostStore", 1,
                                  _ir.NumInputs(1), 0)

    pb_g.createConnectedOp_HostStoreOp({0: t.id}, {}, opid,
                                       ctx._get_op_settings('host_store'),
                                       d2h_stream.tensor_id)
