# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents graphs in the PopART IR."""
from typing import TYPE_CHECKING, Any, Optional, Tuple

import popart._internal.ir as _ir

from popart.ir.tensor import Tensor, Variable, Constant
from popart.ir.globals import pop_current_graph, push_current_graph

if TYPE_CHECKING:
    from popart.ir.ir import Ir

__all__ = ['Graph']


class Graph:
    def __init__(self):
        """The `Graph` class represents graphs in the PopART IR.

        This class cannot be instantiated using the constructor. `Graph`
        instances are created by the `Ir` class. During construction, each `Ir`
        constructs a default `main` graph. More `Graph` instances can be made
        using the `get_graph()` method of the `Ir` class.

        Raises:
            TypeError:
                The `Graph` class cannot be instantiated.
        """
        # The following attributes and their types are declared here for the
        # sake of Python language servers.
        self._ir: Ir = None
        self._debug_name: str = None
        self._pb_graph: _ir.Graph = None

        raise TypeError(f"Cannot create {self.__module__}.Graph instances "
                        "using the constructor.")

    @property
    def name(self) -> str:
        return self._debug_name

    @classmethod
    def _factory(
            cls,
            ir: 'Ir',
            debug_name: str,
            pb_graph: '_ir.Graph',
    ) -> 'Graph':
        """Factory method to construct `Graph` instances.

        Args:
            ir (Ir):
                An instance of `Ir` in which the graph resides.
            debug_name (str):
                A debug name that's assigned to the subgraph.
            pb_graph (_ir.Graph):
                An instance of the low-level pybind11 `Graph`.

        Returns:
            Graph:
                The main graph of the `Ir`.
        """
        self: 'Graph' = super().__new__(cls)
        self._ir = ir
        self._debug_name = debug_name
        self._pb_graph = pb_graph
        return self

    def _create_tensor_id(self, name: Optional[str] = None) -> str:
        """Generate a unique tensor id.

        If the name already exists in the graph it will be modified
        to be made unique in the graph.
        The graph scope will then be added to construct the global unique id.

        Args:
            name (Optional[str]):
                A name which will be appended with an id to make unique.
                Defaults to `t`

        Returns:
            str:
                The unique id of the tensor.
        """
        name = name if name else 't'
        _id = _ir.addScope(self._pb_graph.getScope(), name)
        if _id in self._pb_graph:
            _id = self._ir._pb_ir.createIntermediateTensorId(_id)
        return _id

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, Tensor):
            return value.id in self._pb_graph
        return False

    def get_tensors(self) -> Tuple[Tensor, ...]:
        """Return all Tensors in the Graph"""
        return tuple(
            Tensor._from_pb_tensor(t) for t in self._pb_graph.getTensors())

    def get_variables(self) -> Tuple[Variable, ...]:
        """Return all Variable Tensors in the Graph"""
        return tuple(
            Variable._from_pb_tensor(t)
            for t in self._pb_graph.getTensorsOfType(_ir.TensorType.Variable))

    def get_constants(self) -> Tuple[Constant, ...]:
        """Return all Constant Tensors in the Graph"""
        return tuple(
            Constant._from_pb_tensor(t)
            for t in self._pb_graph.getTensorsOfType(_ir.TensorType.Const))

    def __enter__(self):
        push_current_graph(self)
        return self

    def __exit__(self, *exc):
        pop_current_graph()
        return False
