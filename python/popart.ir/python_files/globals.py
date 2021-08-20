# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from popart.ir.graph import Graph

_CURRENT_GRAPH: List['Graph'] = []


def push_current_graph(g: 'Graph'):
    """Push a graph to the global `_CURRENT_GRAPH` stack.

    Args:
        g (Graph):
          The graph to push.
    """
    global _CURRENT_GRAPH
    _CURRENT_GRAPH.append(g)


def pop_current_graph() -> None:
    """Pop a graph from the global `_CURRENT_GRAPH` stack."""
    global _CURRENT_GRAPH
    _CURRENT_GRAPH.pop()


def get_current_graph() -> 'Graph':
    """Get the graph that is at the top of the global `_CURRENT_GRAPH` stack.

    Raises:
        RuntimeError:
            If the stack is empty.

    Returns:
        Graph:
            The graph at the top of the global `_CURRENT_GRAPH` stack.
    """
    global _CURRENT_GRAPH
    if len(_CURRENT_GRAPH) == 0:
        raise RuntimeError(
            "Trying to access a graph, but no graph has been selected. Hint - "
            "try performing the operations in a context manager (e.g. "
            "`with graph_instance:`) or inside of a function that's called by "
            "`popart.ir.Ir().get_graph()`")
    return _CURRENT_GRAPH[-1]


# Alias for get_current_graph().
gcg = get_current_graph
