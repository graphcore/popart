# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing_extensions import Literal
from popxl import Graph


# pylint: disable=unused-argument
def code_copy(graph: Graph,
              source: Literal['remote', 'io', 'compute'] = 'remote',
              destination: Literal['io', 'compute'] = 'compute'):
    """Copy the graph code from the source location to the destination location.

    There are 3 locations available for a graph at any point:
    * 'remote' means that the graph is located in streaming memory (off-chip).
    * 'io' means it is located on the io tile set (on-chip, but not executable).
    * 'compute' means it is located in executable memory.

    The initial location of the graph will be inferred from the first call to code_copy.
    It is also the user's responsibility to ensure that their subsequent calls
    to code_copy make sense. In particular, if the user never copies a graph to
    the 'compute' location and calls it, it will be assumed to have always been in
    executable memory, and hence always live.

    Calling this function with the same source and destination is disallowed.
    You also cannot copy to streaming memory, since the code is never unloaded
    from streaming memory in the first place.

    Currently 'io' source or destination are not supported either, so you should
    only call this function with the default source and destination values.

    Args:
        graph (Graph): The Graph object to be copied.
        source (Literal['remote', 'io', 'compute']): Location to copy the graph from.
        destination (Literal['io', 'compute']): Location to copy the graph to.
    Raises:
        ValueError: If source and destination are the same.
        NotImplementedError: If the function is called.
    """
    if source == destination:
        raise ValueError(
            f"code_copy called with the same source and destination which is not allowed. Both are assigned to value '{source}'"
        )
    raise NotImplementedError("code_copy function not implemented yet")
