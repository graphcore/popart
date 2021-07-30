# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents the PopART IR."""

from collections import Counter
from typing import Any, Callable

import popart._internal.ir as _ir
from popart.ir.graph import Graph

__all__ = ['Ir']


class Ir:
    """Class that represents the PopART IR.

    This class contains a main graph. Furthermore, it defines methods and
    decorators for creating additional graphs from Python functions.
    """

    def __init__(self):
        """Initializes the `Ir` class."""
        self._pb_ir = _ir.Ir()
        self._main_graph = Graph._factory(self, 'main',
                                          self._pb_ir.getMainGraph())
        self._subgraphs = []
        # Subgraphs are named `{pure_name}_{id}`, where `pure_name` comes from
        # the qualified name of the Python function that created the graph. The
        # following counter counts these so that a unique name can be given to
        # subgraphs that have been created from the same Python function.
        # See `Ir._create_name()`.
        self._pure_names = Counter()

    def main_graph(self) -> 'Graph':
        """Every IR is initialised with a main graph. This method returns this
        graph.

        Returns:
            Graph:
                The main graph of the IR.
        """
        return self._main_graph

    def get_graph(
            self,
            fn: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
    ) -> 'Graph':
        """Create a graph from a Python function.

        Args:
            fn (Callable[..., Any]):
                The Python function that defines the graph.
            *args (Any):
                Arguments passed to the Python function that defines the graph.
            **kwargs (Any):
                Keyword arguments passed to the Python function that defines the
                graph.

        Returns:
            Graph:
                A graph that corresponds to the input Python function.
        """
        name = self._create_name(fn.__qualname__)
        _graph = self._pb_ir.createGraph(name)
        graph = Graph._factory(self, name, _graph)
        self._subgraphs.append(graph)
        return graph

    def _create_name(self, name: str) -> str:
        """Generate a graph name based on the qualified name of the Python
        function that created it.

        Each name will be appended with `_{id}`, where `id` is a positive
        integer, so that if the same function is used to create multiple graphs,
        they will all have unique names.

        NOTE: Occurrences of ".<locals>" in the name are removed.

        Example:
            Suppose a graph function:
                >>> class Foo:
                ...     def bar():
                ...         # Graph definition...
            Creating the following graphs:
                >>> ir.get_graph(Foo.bar)
                >>> ir.get_graph(Foo.bar)
            will result in graph names `Foo.bar_0` and `Foo.bar_1`.

        Args:
            name (str):
                The `__qualname__` attribute of the Python function.

        Returns:
            str:
                The name of the graph.
        """
        name = name.replace(".<locals>", "")
        id = self._pure_names[name]
        self._pure_names[name] += 1
        return f'{name}_{id}'
