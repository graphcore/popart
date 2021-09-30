# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from popart.ir.streams import HostToDeviceStream

from typing import Optional

__all__ = ['host_load']


def host_load(h2d_stream: HostToDeviceStream,
              name: Optional[str] = None) -> Tensor:
    """
    Host Load Op: an op to represent the transfer of data from the host to the
    device. It uses the existing host to device transfers created when building
    the IR, but defers the actual poplar::Copy until the op itself runs. This
    allows the copy to be scheduled as part of the normal op scheduling.


    Args:
        dtype (dtypes.dtype): Data type for the output Tensor
        shape (Tuple[int]): Shape of the output tensor.
        name (str): Name to use for the poplar stream.

    Returns:
        Tensor: The output tensor streamed from host.
    """
    ctx = get_current_context()
    g = ctx.graph

    dtype = h2d_stream.dtype
    shape = h2d_stream.shape
    stream_tensor_id = h2d_stream.tensor_id()

    if name is None:
        pb_main = g.ir().main_graph()._pb_graph
        name = _ir.removeScope(pb_main, stream_tensor_id)

    name_init = g._create_tensor_id(name + '_init')
    name_hostload = g._create_tensor_id(name + '_hostload')

    pb_g = g._pb_graph
    info = _ir.TensorInfo(dtype._pb_dtype, shape)

    opid_init = _ir.OperatorIdentifier("ai.graphcore", "Init", 1,
                                       _ir.NumInputs(0), 1)

    pb_g.createConnectedOp_InitOp(
        {},
        {0: name_init},
        opid_init,
        info,
        _ir.TensorType.ActGrad,
        _ir.InitType.Zero,
        ctx._get_op_settings('init'),
        -1,
    )

    opid_host_load = _ir.OperatorIdentifier("ai.graphcore", "HostLoad", 1,
                                            _ir.NumInputs(1), 1)

    pb_g.createConnectedOp_HostLoadOp(
        {0: name_init},
        {0: name_hostload},
        opid_host_load,
        ctx._get_op_settings('host_load'),
        stream_tensor_id,
    )

    return Tensor._from_pb_tensor(pb_g.getTensor(name_hostload))
