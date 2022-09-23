# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart import ScatterReduction
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from typing_extensions import Literal
from .utils import (
    check_in_graph,
    convert_optional_float,
    check_tensor_ipu_and_tile_set,
)

ScatterReductionType = Literal["sum", "min", "max", "none"]

reduction_dict = {
    "sum": ScatterReduction.Sum,
    "min": ScatterReduction.Min,
    "max": ScatterReduction.Max,
    "none": ScatterReduction.NoReduction,
}


@op_debug_context
def scatter_reduce(
    data: Tensor,
    indices: Tensor,
    reduction: ScatterReductionType,
    initial_values: Optional[Tensor] = None,
    axis: int = 0,
    axis_size: Optional[int] = None,
    available_memory_proportion: Optional[float] = None,
) -> Tensor:

    if initial_values is None and axis_size is None:
        raise ValueError("One of initial_values and axis_size must be specified")

    if axis_size is None:
        axis_size = initial_values.shape[axis]

    try:
        reduction_type = reduction_dict[reduction]
    except KeyError:
        raise ValueError(f"The reduction must be one of {list(reduction_dict.keys())}")

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    if initial_values is None:
        tensor_args = {"data": data, "indices": indices}
        op_inputs = {0: data.id, 1: indices.id}
    else:
        tensor_args = {
            "data": data,
            "indices": indices,
            "initial_values": initial_values,
        }
        op_inputs = {0: data.id, 1: indices.id, 2: initial_values.id}

    check_in_graph(g, **tensor_args)
    check_tensor_ipu_and_tile_set(**tensor_args)

    available_memory_proportion = convert_optional_float(available_memory_proportion)

    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "ScatterReduce", 1, _ir.NumInputs(2, 3), 1
    )
    settings = ctx._get_op_settings("scatterreduce")
    op = pb_g.createConnectedOp_ScatterReduceOp(
        op_inputs,
        {0: g._create_tensor_id("scatterreduce_out")},
        opid=opid,
        axis_=axis,
        axis_size_=axis_size,
        available_memory_proportion_=available_memory_proportion,
        reduction_=reduction_type,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
