# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents the PopART IR."""

from collections import Counter
from typing import Any, Callable

import popart._internal.ir as _ir
from popart.ir.graph import Graph
from popart.ir.context import gcg
from popart.ir.module import Module
from popart.ir.tensor import Tensor, subgraph_input, subgraph_output

__all__ = ['Ir']


class Ir:
    """Class that represents the PopART IR.

    This class contains a main graph. Furthermore, it defines methods and
    decorators for creating additional graphs from Python functions.
    """

    def __init__(self):
        """Initialises a new `Ir`."""
        self._pb_ir = _ir.Ir()

    @classmethod
    def _from_pb(
            cls,
            pb_ir: '_ir.Ir',
    ) -> 'Ir':
        """Factory method to construct `Ir` instances.

        Args:
            pb_ir (_ir.Ir):
                An instance of the low-level pybind11 `Ir`.

        Returns:
            Ir:
                A popart.ir.Ir that reprsents the passed pb_ir.
        """
        self: 'Ir' = super().__new__(cls)
        self._pb_ir = pb_ir
        return self

    def main_graph(self) -> 'Graph':
        """Every IR is initialised with a main graph. This method returns this
        graph.

        Returns:
            Graph:
                The main graph of the IR.
        """
        return Graph._from_pb(self._pb_ir.getMainGraph())

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
        g = gcg()
        pb_g = g._pb_graph

        if isinstance(fn, Module):
            qualname = fn.__class__.__qualname__
        else:
            # Note all Python functions will have __qualname__.
            if not callable(fn) or not hasattr(fn, '__qualname__'):
                raise TypeError(
                    "Callable `fn` must be either a function or a class that extends popart.ir.Module"
                )
            else:
                qualname = fn.__qualname__

        name = self._create_name(qualname)
        _pb_subgraph = self._pb_ir.createGraph(name)
        subgraph = Graph._from_pb(_pb_subgraph)

        with subgraph:
            # FIXME: Ignore/warn/error on duplicate inputs, as we do not want
            # to create dubplicate subgraph inputs

            in_args = []
            for arg in args:
                if isinstance(arg, Tensor):
                    t = arg
                    in_args.append(
                        subgraph_input(t.dtype, t.shape,
                                       _ir.removeScope(pb_g, t.id)))
                else:
                    in_args.append(arg)

            in_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, Tensor):
                    t = v
                    in_kwargs[k] = subgraph_input(t.dtype, t.shape,
                                                  _ir.removeScope(pb_g, t.id))
                else:
                    in_kwargs[k] = v

            outputs = fn(*in_args, **in_kwargs)

            if outputs is None:
                outputs = []

            if isinstance(outputs, Tensor):
                outputs = (outputs, )

            for out in outputs:
                subgraph_output(out)

        return subgraph

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
        name = self._pb_ir.createUniqueSubgraphId(name)
        return name
