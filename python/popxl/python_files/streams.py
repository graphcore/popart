# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popxl.context import gcg, gmg
from popxl.tensor import Tensor, TensorSpec
from popxl.dtypes import dtype

import popart._internal.ir as _ir

from typing import Any, Iterable, Optional, Tuple


class Stream:
    def __init__(self):
        raise NotImplementedError(
            "Cannot construct a popxl.Stream directly. Use `h2d_stream` or `d2h_stream`."
        )

    @classmethod
    def _from_tensor(cls, tensor: Tensor) -> Tensor:
        """Construct Stream object from a tensor."""
        self = super().__new__(cls)
        self._stream_tensor = tensor
        return self

    @property
    def dtype(self) -> dtype:
        """Return the type of the stream."""
        return self._stream_tensor.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return a tuple representing the shape of the stream."""
        return self._stream_tensor.shape

    @property
    def spec(self) -> TensorSpec:
        """Return a TensorSpec instance using properties of the stream."""
        return self._stream_tensor.spec

    @property
    def tensor_id(self) -> str:
        """Return the identifier of the stream."""
        return self._stream_tensor.id

    def __hash__(self) -> int:
        """Calculate a hash."""
        return hash(self._stream_tensor)

    def __eq__(self, other: Any) -> bool:
        """Perform an equality check."""
        return isinstance(
            other, Stream) and self._stream_tensor == other._stream_tensor

    def __str__(self) -> str:
        """Return a string representation."""
        return str(self._stream_tensor)


class HostToDeviceStream(Stream):
    """
    A host-to-device stream in the IR.

    This can be created in the main graph of the IR only.

    You can pass a host-to-device stream and a tensor to `popxl.ops.host_load()` in
    any subgraph any number of times, and in all cases PopART will stream the
    next value into the tensor.
    """

    def __str__(self):
        return f"HostToDeviceStream {super().__str__()}"


class DeviceToHostStream(Stream):
    """
    A device-to-host stream in the IR.

    This can be created in the main graph of the IR only.

    You can pass a device-to-host stream and a tensor to `popxl.ops.host_store()` in
    any subgraph any number of times, and in all cases PopART will send the
    tensor to the stream.
    """

    def __str__(self):
        return f"DeviceToHostStream {super().__str__()}"


def h2d_stream(shape: Iterable[int], dtype: dtype,
               name: Optional[str] = None) -> HostToDeviceStream:
    g = gcg()
    mg = gmg()

    if g.name != mg.name:
        raise ValueError(
            "popxl: Can only call `h2d_stream` in context of main graph. You are in context of graph:",
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
    mg = gmg()
    ir_ = mg.ir._pb_ir

    if g.name != mg.name:
        raise ValueError(
            "popxl: Can only call `d2h_stream` in context of main graph. You are in context of graph:",
            g.name)

    pb_mg = mg._pb_graph

    if name is None:
        name = "d2h_stream"
    name = mg._create_tensor_id(name)

    pb_mg.addActGrad(name)
    pb_t = pb_mg.getTensor(name)
    pb_t.info = _ir.TensorInfo(dtype._pb_dtype, list(shape))
    ir_.addAnchor(pb_t.id)

    return DeviceToHostStream._from_tensor(Tensor._from_pb_tensor(pb_t))
