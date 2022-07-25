# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Decompose Gradients Sum transform."""
import popart._internal.ir as _ir
from popxl.graph import Graph


def decompose_sum(graph: Graph) -> Graph:
    """
    Transform the input Graph by decomposing Sum operations with >2 inputs into a liveness optimal tree of additions.

    Args:
        graph (Graph):
            The graph to decompose sum.

    Returns:
        decomposed_graph (:py:class:`popxl.Graph`): decomposed gradient sum graph.
    """
    transform = _ir.transforms.DecomposeSum()
    transform.apply(graph._pb_graph)
    return graph
