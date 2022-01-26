# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional

import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from popart.ir.streams import HostToDeviceStream
from .init import init


@op_debug_context
def host_load(h2d_stream: HostToDeviceStream,
              name: Optional[str] = None) -> Tensor:
    """
    Host Load Op: an op to represent the transfer of data from the host to the
    device. It uses the existing host to device transfers created when building
    the IR, but defers the actual poplar::Copy until the op itself runs. This
    allows the copy to be scheduled as part of the normal op scheduling.


    Args:
        h2d_stream: (HostToDeviceStream) Stream to load from.
        name (str): Name to use for the returned tensor.

    Returns:
        Tensor: The output tensor streamed from host.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    shape = h2d_stream.shape
    dtype = h2d_stream.dtype
    stream_tensor_id = h2d_stream.tensor_id()

    if name is None:
        pb_main = g.ir().main_graph()._pb_graph
        name = _ir.removeScope(pb_main, stream_tensor_id)

    init_tensor = init(shape, dtype, name + '_init')

    name_hostload = g._create_tensor_id(name + '_hostload')

    opid = _ir.OperatorIdentifier("ai.graphcore", "HostLoad", 1,
                                  _ir.NumInputs(1), 1)
    pb_g.createConnectedOp_HostLoadOp(
        {0: init_tensor.id},
        {0: name_hostload},
        opid,
        ctx._get_op_settings('host_load'),
        stream_tensor_id,
    )

    return Tensor._from_pb_tensor(pb_g.getTensor(name_hostload))
