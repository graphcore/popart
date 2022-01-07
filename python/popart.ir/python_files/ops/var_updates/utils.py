# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Any, Dict
import popart._internal.ir as _ir
from popart.ir.graph import Graph
from popart.ir.tensor import Tensor
from ..utils import check_in_graph


def handle_optimizer_value(g: Graph, f: Any, ins: Dict[int, str], index: int):
    if isinstance(f, Tensor):
        check_in_graph(g, f=f)
        ins[index] = f.id
        return _ir.OptimizerValue(0.0, False)
    elif f is None:
        return _ir.OptimizerValue()

    return _ir.OptimizerValue(f, True)
