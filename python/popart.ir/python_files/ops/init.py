# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Optional

import popart._internal.ir as _ir
from popart.ir import dtypes, Tensor
from popart.ir.context import get_current_context, op_debug_context

__all__ = ['init']


@op_debug_context
def init(shape: Iterable[int], dtype: dtypes.dtype,
         name: Optional[str] = None) -> Tensor:
    """
    Init Op: create a tensor with zero values.
        The returned tensor is not considered a variable.

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
