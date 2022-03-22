# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Tuple
import popart._internal.ir as _ir
from popxl.context import debug_context_frame_offset, get_current_context, op_debug_context
from popxl.tensor import Tensor, constant
from popxl.dtypes import uint32
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def create_random_seed(seed: Tensor, modifier: Tensor) -> Tensor:
    """Create a new random seed.

    Args:
        seed (Tensor): Seed Tensor used to produce the new seed. Must be shape=(2,) dtype=uint32.
        modifier (Tensor): A scalar modifier to be combined with `seed` before generating the output.

    Raises:
        ValueError: If the seed Tensor does not have shape=(2,) or dtype=uint32.

    Returns:
        Tensor: A new random seed.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, seed=seed, modifier=modifier)
    check_tensor_ipu_and_tile_set(seed=seed, modifier=modifier)

    if seed.shape != (2, ) or seed.dtype != uint32:
        raise ValueError(
            f"seed Tensor must be shape=(2,) dtype=uint32. Provided {seed}")

    settings = ctx._get_op_settings('modify_seed')
    opid = _ir.OperatorIdentifier("ai.graphcore", "ModifyRandomSeed", 1,
                                  _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_ModifyRandomSeedOp(
        {
            0: seed.id,
            1: modifier.id
        },
        {
            0: g._create_tensor_id("modified_seed"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@debug_context_frame_offset(1)
def split_random_seed(seed: Tensor, n: int = 2) -> Tuple[Tensor, ...]:
    """
    Produce `n` random seeds from a seed.

    Chaining calls to `split_random_seed` can be used to ensure unique random behaviour
    across a program. For example:

    .. code-block:: python

       seed, s1 = ops.split_random_seed(seed)
       y = ops.dropout(x, s1)
       seed, s2 = ops.split_random_seed(seed)
       z = ops.dropout(y, s2)

    Args:
        seed (Tensor): Seed tensor used to be produce new seeds. Must be shape=(2,) dtype=uint32.
        n (int, optional): Number of new seeds to produce. Defaults to 2.

    Returns:
        Tuple[Tensor, ...]: New seeds
    """
    return tuple(
        create_random_seed(seed, constant(i, uint32)) for i in range(n))
