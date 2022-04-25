# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, Optional, Union
from typing_extensions import Literal

import popart._internal.ir as _ir
from popxl import dtypes, Tensor
from popxl.context import get_current_context, op_debug_context


@op_debug_context
def init(shape: Iterable[int],
         dtype: dtypes.dtype,
         name: Optional[str] = None,
         init_type: Union[Literal["zero"], Literal["undef"]] = "zero"
         ) -> Tensor:
    """
    Create a tensor that is initialised with zero or undefined values.

    The returned tensor is not considered a variable.
    A variable must be created in the main graph; it can be initialised to arbitrary values and can be read/written with session methods.

    In contrast, `init` can be executed anywhere so it can return an initialised tensor in non-main graphs.

    The tensor can only be initialised to zero or undefined values.

    Args:
        dtype (dtypes.dtype): Data type of the output tensor.
        shape (Tuple[int]): Shape of the output tensor.
        name (str): Name of the output tensor.
        init_type (Union[Literal["zero"], Literal["undef"]]): Initialisation of the output tensor.

    Raises:
        ValueError: If the `init_type` is unknown.

    Returns:
        Tensor: An initialised tensor.
    """
    ctx = get_current_context()
    g = ctx.graph

    pb_g = g._pb_graph
    info = _ir.TensorInfo(dtype._pb_dtype, list(shape))

    if init_type == "zero":
        pb_init_type = _ir.InitType.Zero
    elif init_type == "undef":
        pb_init_type = _ir.InitType.NoInit
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    opid_init = _ir.OperatorIdentifier("ai.graphcore", "Init", 1,
                                       _ir.NumInputs(0), 1)
    op = pb_g.createConnectedOp_InitOp(
        {},
        {0: g._create_tensor_id(name)},
        opid_init,
        info,
        _ir.TensorType.ActGrad,
        pb_init_type,
        ctx._get_op_settings('init'),
        -1,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
