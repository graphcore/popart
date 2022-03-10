# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from popxl.remote_buffer import RemoteBuffer
from popxl import constant
from popxl.dtypes import uint32
from .utils import check_in_graph
from .init import init


@op_debug_context
def remote_load(remote_buffer: RemoteBuffer,
                offset: Union[int, Tensor],
                name: Optional[str] = None) -> Tensor:
    """
    Load a tensor from Streaming Memory.

    This operation loads a tensor from the remote buffer residing in the off-chip Streaming Memory.

    The tensor will be loaded from the memory location corresponding to
    ``remote_buffer_id`` (specified in ``remote_buffer``).

    The relationship between ``offset`` and ``remote_buffer_id`` is thoroughly
    described in ``remote_store``.

    Note:
        There is no data dependency (in the graph) between remote store and remote load.
        Thus, the remote load operator may end up before the remote store operator in the
        serialized graph.
        One way to circumvent this is by using ``with popxl.in_sequence(True)``

    See also:
        ``remote_buffer``, ``remote_store``, ``remote_load_``

    Args:
        remote_buffer (RemoteBuffer): The handle to the remote buffer
        offset (Union[int, Tensor]): Integer or rank-0 tensor indicating what row/entry in the
          remote buffer to load from
        name (str): Name to use for the returned tensor

    Returns:
        Tensor: The tensor loaded from the remote buffer
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    shape = remote_buffer.tensor_shape
    dtype = remote_buffer.tensor_dtype
    remote_buffer_id = remote_buffer.remote_buffer_id

    # Create tensors
    if isinstance(offset, int):
        offset = constant(offset, uint32, name="offset")

    if name is None:
        name = f"id_{remote_buffer_id}_offset_{offset.name}"

    remote_load_tensor = init(shape, dtype, name + '_remote_load', 'undef')

    check_in_graph(g, remote_load_tensor=remote_load_tensor, offset=offset)

    # Set the meta_shape of the input tensor. Required for RTS.
    info = remote_load_tensor._pb_tensor.info
    info.set(info.dataType(), info.shape(), remote_buffer.meta_shape)

    remote_buffer.validate_tensor_matches_buffer(remote_load_tensor)

    settings = ctx._get_op_settings('remote_load')
    opid = _ir.OperatorIdentifier("ai.graphcore", "RemoteLoad", 1,
                                  _ir.NumInputs(1, 2), 1)

    op = pb_g.createConnectedOp_RemoteLoadOp(
        {
            0: remote_load_tensor.id,
            1: offset.id
        }, {0: g._create_tensor_id(name + "_remote_loaded")}, opid, settings,
        remote_buffer.remote_buffer_id)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def remote_load_(remote_buffer: RemoteBuffer, offset: Union[int, Tensor],
                 t: Tensor) -> Tensor:
    """
    Load a tensor from Streaming Memory (in-place).

    This operation loads a tensor in-place from the remote buffer residing in the off-chip Streaming Memory.

    This op is identical to ``remote_load``, but with the exception that the tensor loaded from
    the remote buffer will be written to ``t`` directly.

    Note:
        There is no data dependency (in the graph) between remote store and remote load.
        Thus, the remote load operator may end up before the remote store operator in the
        serialized graph.
        One way to circumvent this is by using ``with popxl.in_sequence(True)``

    See also:
        ``remote_buffer``, ``remote_store``, ``remote_load``

    Args:
        remote_buffer (RemoteBuffer): The handle to the remote buffer
        offset (Union[int, Tensor]): Integer or rank-0 tensor indicating what row/entry in the
          remote buffer to load from
        t (Tensor): The tensor the loaded data will written to

    Returns:
        Tensor: The tensor loaded from the remote buffer
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    # Create tensors
    if isinstance(offset, int):
        offset = constant(offset, uint32, name="offset")

    check_in_graph(g, t=t, offset=offset)

    # Set the meta_shape of the input tensor. Required for RTS.
    info = t._pb_tensor.info
    info.set(info.dataType(), info.shape(), remote_buffer.meta_shape)

    remote_buffer.validate_tensor_matches_buffer(t)

    settings = ctx._get_op_settings('remote_load_inplace')
    opid = _ir.OperatorIdentifier("ai.graphcore", "RemoteLoadInplace", 1,
                                  _ir.NumInputs(1, 2), 1)

    op = pb_g.createConnectedOp_RemoteLoadInplaceOp(
        {
            0: t.id,
            1: offset.id
        }, {0: g._create_tensor_id("remote_load_inplace_out")}, opid, settings,
        remote_buffer.remote_buffer_id)

    return Tensor._from_pb_tensor(op.outTensor(0))


def multiply_tuple(tup: Tuple) -> int:
    """Multiply the elements in a tuple

    Args:
        tup (Tuple): The tuple.

    Returns:
        int: The product of the elements.
    """
    temp = list(tup)
    product = 1
    for x in temp:
        product *= x
    return product
