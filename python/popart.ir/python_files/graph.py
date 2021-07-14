from typing import TYPE_CHECKING, Optional

import popart._internal.ir as _ir
from popart.ir.globals import pop_current_graph, push_current_graph
from popart.ir.tensor import Tensors
from popart.util.ast import parse_assign_from_ast, parse_mutli_assign_from_ast

if TYPE_CHECKING:
    from popart.ir.ir import Ir


class Graph:
    def __init__(self, ir, debug_name) -> None:
        self._tensors = []
        self._inputs = []
        self._outputs = []
        self._ir = ir
        if debug_name is None:
            debug_name = ir.get_next_subgraph_uid()
        self._graph = _ir.Graph(ir._ir, debug_name)

    # @property
    # def id(self) -> str:
    #     return self._debug_name

    def create_op(self, inputs: Optional[Tensors], debug_name: Optional[str]):
        pass

    def __enter__(self):
        push_current_graph(self)

    def __exit__(self, *_):
        pop_current_graph()
