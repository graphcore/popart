# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popart.ir.tensor import Tensor
from popart.ir.graph import Graph
import popart._internal.ir as _ir
from typing import Optional


def check_in_graph(graph: Graph, *tensors: Tensor):
    """Checks if tensors are in graph. If not, raises a ValueError."""
    for tensor in tensors:
        if tensor not in graph:
            raise ValueError(
                f"Tensor {tensor.name} is not in the current Graph {graph.name}."
            )


def convert_optional_float(v: Optional[float]):
    return _ir.OptionalFloat(v) if v is not None else _ir.OptionalFloat()
