# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popart.ir.context import gcg
from popart.ir.tensor import Tensor
from popart.ir.dtypes import dtype

import popart._internal.ir as _ir

from typing import Any, Iterable, Optional, Tuple

__all__ = [
    'HostToDeviceStream', 'DeviceToHostStream', 'h2d_stream', 'd2h_stream'
]


class _Stream:
    def __init__(self):
        raise NotImplementedError(
            "Cannot construct a popart.ir._Stream directly.")

    @classmethod
    def _from_tensor(cls, tensor: Tensor):
        self = super().__new__(cls)
        self._stream_tensor = tensor
        return self

    @property
    def dtype(self) -> dtype:
        return self._stream_tensor.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._stream_tensor.shape

    def tensor_id(self) -> str:
        return self._stream_tensor.id

    def __hash__(self):
        return hash(self._stream_tensor)

    def __eq__(self, other: Any):
        return isinstance(
            other, _Stream) and self._stream_tensor == other._stream_tensor

    def __str__(self):
        return str(self._stream_tensor)


class HostToDeviceStream(_Stream):
    """
    A host-to-device stream in the Ir.

    Can be created in the main graph of the Ir only.

    You can pass a HostToDeviceStream and a Tensor to ops.host_load in any
    subgraph(s) any number of times, and in all cases PopART will stream the next
    value into the provided tensor.
    """

    def __str__(self):
        return f"HostToDeviceStream {super().__str__()}"


class DeviceToHostStream(_Stream):
    """
    A device-to-host stream in the Ir.

    Can be created in the main graph of the Ir only.

    You can pass a DeviceToHostStream and a Tensor to ops.host_store in any
    subgraph(s) any number of times, and in all cases PopART will stream the value
    of the provided tensor down the provided stream.
    """

    def __str__(self):
        return f"DeviceToHostStream {super().__str__()}"


def h2d_stream(shape: Iterable[int], dtype: dtype,
               name: Optional[str] = None) -> HostToDeviceStream:
    g = gcg()
    mg = g.ir().main_graph()

    if g.name != mg.name:
        raise ValueError(
            "popart.ir: Can only call `h2d_stream` in context of main graph. You are in context of graph:",
            g.name)

    pb_mg = mg._pb_graph

    if name is None:
        name = "h2d_stream"
    name = mg._create_tensor_id(name)

    pb_mg.addStream(name, _ir.TensorInfo(dtype._pb_dtype, list(shape)), name)

    return HostToDeviceStream._from_tensor(
        Tensor._from_pb_tensor(pb_mg.getTensor(name)))


def d2h_stream(shape: Iterable[int], dtype: dtype,
               name: Optional[str] = None) -> DeviceToHostStream:
    g = gcg()
    mg = g.ir().main_graph()

    if g.name != mg.name:
        raise ValueError(
            "popart.ir: Can only call `d2h_stream` in context of main graph. You are in context of graph:",
            g.name)

    pb_mg = mg._pb_graph

    if name is None:
        name = "d2h_stream"
    name = mg._create_tensor_id(name)

    pb_mg.addActGrad(name)
    pb_t = pb_mg.getTensor(name)
    pb_t.info = _ir.TensorInfo(dtype._pb_dtype, list(shape))

    return DeviceToHostStream._from_tensor(Tensor._from_pb_tensor(pb_t))
