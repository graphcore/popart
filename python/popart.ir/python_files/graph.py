# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents graphs in the PopART IR."""

from collections import Counter
from typing import TYPE_CHECKING

import popart._internal.ir as _ir
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
        self._pure_tensor_names: Counter = None

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
        # Tensors are named `{pure_name}_{id}`, where `pure_name` is either
        # user-specified or inferred from the name of the Python variable that a
        # tensor is assigned to. The following counter counts these so that a
        # unique name can be given to tensors that have the same pure name.
        # See `Graph._create_name()`.
        self._pure_tensor_names = Counter()
        return self

    def _create_tensor_name(self, name: str) -> str:
        """Generate a unique tensor name.

        Each name will be appended with `_{id}`, where `id` is a positive
        integer, so that all tensors within a graph have unique names.

        Args:
            name (str):
                A name which will be appended with an id to make unique.

        Returns:
            str:
                The unique name of the tensor.
        """
        id = self._pure_tensor_names[name]
        self._pure_tensor_names[name] += 1
        return f'{name}_{id}'

    def __enter__(self):
        push_current_graph(self)
        return self

    def __exit__(self, *exc):
        pop_current_graph()
        return False
