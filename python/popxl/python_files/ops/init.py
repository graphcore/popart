# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Optional

import popart._internal.ir as _ir
from popxl import dtypes, Tensor
from popxl.context import get_current_context, op_debug_context


@op_debug_context
def init(shape: Iterable[int], dtype: dtypes.dtype,
         name: Optional[str] = None) -> Tensor:
    """
    Create a tensor that is initialised with zero or undefined values.

    The returned tensor is not considered a variable.
    Variable must be created in the main_graph, can be initialised to arbitrary values and can be read/written to with session methods.
    In contrast, `init` can be executed anywhere so it can return an initialised tensor in non-main graphs. However, it can only be initialised to zero or undefined values.

    Args:
        dtype (dtypes.dtype): Data type for the output Tensor
        shape (Tuple[int]): Shape of the output tensor.
        name (str): Name to use for the poplar stream.

    Returns:
        Tensor: The output tensor streamed from host.
    """
    ctx = get_current_context()
    g = ctx.graph

    pb_g = g._pb_graph
    info = _ir.TensorInfo(dtype._pb_dtype, list(shape))

    opid_init = _ir.OperatorIdentifier("ai.graphcore", "Init", 1,
                                       _ir.NumInputs(0), 1)
    op = pb_g.createConnectedOp_InitOp(
        {},
        {0: g._create_tensor_id(name)},
        opid_init,
        info,
        _ir.TensorType.ActGrad,
        _ir.InitType.Zero,
        ctx._get_op_settings('init'),
        -1,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
