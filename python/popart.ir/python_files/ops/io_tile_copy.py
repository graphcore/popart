# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def io_tile_copy(t: Tensor) -> Tensor:
    """
    Copies a tensor to/from IO tiles on the current IPU.

    Args:
        t: Tensor
            Tensor to be copied.

    Returns:
        t_copied: Tensor
            The copied tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('iotilecopy')

    # Use internal method to infer the input tensor's tileSet
    vgid, tile_set = t._pb_tensor.getVirtualGraphIdAndTileSetUnsafe()
    if vgid != -1:
        settings.vgraphId = _ir.OptionalVGraphId(vgid)
    if tile_set != _ir.TileSet.Undefined:
        # TileSet should match the destination
        # so it should be the opposite of the source `t`.
        settings.tileSet = _ir.TileSet.IO if tile_set == _ir.TileSet.Compute else _ir.TileSet.Compute

    opid = _ir.OperatorIdentifier("ai.graphcore", "IoTileCopy", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_IoTileCopyOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id(t.name + f"_iotilecopy"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
