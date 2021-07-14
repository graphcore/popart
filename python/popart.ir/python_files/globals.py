from typing import List, TYPE_CHECKING

_CURRENT_GRAPH = []


def push_current_graph(g):
    global _CURRENT_GRAPH
    _CURRENT_GRAPH.append(g)


def pop_current_graph():
    global _CURRENT_GRAPH
    _CURRENT_GRAPH.pop()


def get_current_graph():
    global _CURRENT_GRAPH
    if len(_CURRENT_GRAPH) == 0:
        raise RuntimeError("")
    return _CURRENT_GRAPH[-1]


gcg = get_current_graph
