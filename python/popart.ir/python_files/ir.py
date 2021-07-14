# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import inspect
from typing import Any, Callable, List

import popart._internal.ir as _ir
from popart.ir.graph import Graph
from popart.ir.tensor import Tensor, TensorInfo
from popart.ir.util import tidy_name


# NOTE: This class is a placeholder to allow us to test access to
# popart._internal._ir and may not actually be required to implement popart.ir.
# If this class is not required it can be removed.
class Ir:
    """ Class that represents an IR to the popart.ir user. """

    def __init__(self):
        self._ir = _ir.Ir()
        self._main_graph = Graph(self, "main_graph")
        self._graphs = [self._main_graph]

    def main_graph(self) -> Graph:
        return self._main_graph

    def get_graph(
            self,
            fn: Callable[..., Any],
            *inputs_infos: List[TensorInfo],
    ):
        inputs_names = inspect.signature(fn).parameters.keys()
        if len(inputs_infos) != len(inputs_names):
            raise RuntimeError("Wrong number of inputs")

        graph = Graph(self, tidy_name(fn.__qualname__))
        self._graphs.append(graph)
        with graph:
            inputs = [
                Tensor(info.dtype, info.shape, debug_name=name)
                for name, info in zip(inputs_names, inputs_infos)
            ]
            outputs = fn(*inputs)
            print(outputs)

    def get_next_subgraph_uid(self):
        uid = 0
        while True:
            yield f'subgraph_{uid}'
            uid += 1
