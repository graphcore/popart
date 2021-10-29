# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional

import popart._internal.ir as _ir
from popart.ir.tensor import Tensor
from popart.ir.graph import Graph
from popart.ir.dtypes import dtype


def cast_if_needed(t: Tensor, data_type: dtype) -> Tensor:
    from popart.ir.ops.cast import cast
    if t.dtype != data_type:
        return cast(t, data_type)
    return t


def check_in_graph(graph: Graph, *tensors: Tensor):
    """Checks if tensors are in graph. If not, raises a ValueError."""
    for tensor in tensors:
        if tensor not in graph:
            raise ValueError(
                f"{tensor} is not in the current Graph {graph.name}.")


def handle_negative_axis(t: Tensor, axis: int) -> int:
    return len(t.shape) + axis if axis < 0 else axis


def convert_optional_float(v: Optional[float]):
    return _ir.OptionalFloat(v) if v is not None else _ir.OptionalFloat()


def convert_optional_dtype(dt: Optional[dtype]):
    return _ir.OptionalDataType(
        dt._pb_dtype) if dt is not None else _ir.OptionalDataType()
