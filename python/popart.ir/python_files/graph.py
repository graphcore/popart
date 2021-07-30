# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents graphs in the PopART IR."""

from typing import TYPE_CHECKING

import popart._internal.ir as _ir

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
