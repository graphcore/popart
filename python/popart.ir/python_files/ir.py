# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents the PopART IR."""

from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional
from weakref import WeakValueDictionary

import popart
import popart._internal.ir as _ir
from popart.ir.graph import Graph
from popart.ir.module import Module
from popart.ir.tensor import Tensor, subgraph_input, subgraph_output

if TYPE_CHECKING:
    IrCache = WeakValueDictionary[int, 'Ir']

__all__ = ['Ir']


class Ir:
    """
    Class that represents the PopART IR.

    This class contains a main graph. Furthermore, it defines methods and
    decorators for creating additional graphs from Python functions.
    """

    def __init__(self):
        """Initialises a new `Ir`."""
        self._pb_ir = _ir.Ir()
        # Set better defaults for popart.ir programs.
        # Some parts of graph construction use the session options to make decisions,
        # such as inheritPlacementAttributes and scheduling priorities. So we must set
        # the options when we construct the _internal IR before the user starts constructing
        # the graphs.
        opts = self._pb_ir.getSessionOptions()
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.useHostCopyOps = True
        opts.aliasZeroCopy = True
        opts.enableExplicitMainLoops = True
        opts.explicitRecomputation = True

        Ir._ir_cache[self.id] = self
        self._graph_cache: Dict[str, Graph] = {}

    _ir_cache: 'IrCache' = WeakValueDictionary()

    @classmethod
    def _from_pb(
            cls,
            pb_ir: '_ir.Ir',
    ) -> 'Ir':
        """Factory method to return popart.ir.Ir for a pybind _ir.Ir.

        Args:
            pb_ir (_ir.Ir):
                An instance of the low-level pybind11 `Ir`.

        Raises:
            RuntimeError: If the pb_ir does not already have a popart.ir Ir in-memory.

        Returns:
            Ir:
                A popart.ir.Ir that reprsents the passed pb_ir.
        """
        _id = pb_ir.getId()
        if _id not in Ir._ir_cache:
            raise RuntimeError(
                "Constructing a new Ir with _from_pb is unexpected. "
                "This implies the Ir was garbage collected. "
                "Is a popart.ir class missing a back reference to Ir?")
        return Ir._ir_cache[_id]

    def main_graph(self) -> 'Graph':
        """Every IR is initialised with a main graph. This method returns this
        graph.

        Returns:
            Graph:
                The main graph of the IR.
        """
        return Graph._from_pb(self._pb_ir.getMainGraph())

    def create_graph(
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
            dupes = self._find_duplicate_tensors(
                [arg for arg in args if isinstance(arg, Tensor)])
            if len(dupes) > 0:
                raise ValueError(
                    f"There are duplicate tensors in the inputs to {name}: {[d.id for d in dupes]} "
                    "Please ensure tensor inputs to subgraphs are unique.")

            in_args = []
            for arg in args:
                if isinstance(arg, Tensor):
                    t = arg
                    in_args.append(subgraph_input(t.shape, t.dtype, t.name))
                else:
                    in_args.append(arg)

            in_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, Tensor):
                    t = v
                    in_kwargs[k] = subgraph_input(t.shape, t.dtype, t.name)
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

    def _find_duplicate_tensors(self,
                                tensors: Iterable[Tensor]) -> List[Tensor]:
        """Get the duplicate tensors (by tensor id) from a list.

        Args:
            tensors (Iterable[Tensor]): Tensors to check

        Returns:
            List[Tensor]: The duplicates (once each).
        """
        seen = {}
        dupes = []

        for t in tensors:
            if t.id not in seen:
                seen[t.id] = 1
            else:
                if seen[t.id] == 1:
                    dupes.append(t)
                seen[t.id] += 1

        return dupes

    def create_empty_graph(self, name: Optional[str] = None):
        """Create a new graph.

        Args:
            name (Optional[str], optional): Name of the graph. Defaults to "graph".

        Returns:
            Graph
        """
        name = name or "graph"
        _pb_subgraph = self._pb_ir.createGraph(
            name)  # type: ignore GraphId != str
        return Graph._from_pb(_pb_subgraph)

    def _create_name(self, name: str) -> str:
        """Generate a graph name based on the qualified name of the Python
        function that created it.

        NOTE: Occurrences of ".<locals>" in the name are removed.

        Example:
            Suppose a graph function:
                >>> class Foo:
                ...     def bar():
                ...         # Graph definition...
            Creating the following graphs:
                >>> ir.create_graph(Foo.bar)
                >>> ir.create_graph(Foo.bar)
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

    @property
    def id(self) -> int:
        return self._pb_ir.getId()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Ir) and self.id == other.id

    def __repr__(self) -> str:
        return f"Ir[id={self.id}]"
