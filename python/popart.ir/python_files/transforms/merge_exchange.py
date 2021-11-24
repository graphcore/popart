# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Set
from contextlib import contextmanager
import popart._internal.ir as _ir
from popart.ir.context import get_current_context

__all__ = ["merge_exchange"]


@contextmanager
def merge_exchange():
    """Combine RemoteLoad/RemoteStore/HostLoad/HostStore operations into a single MergeExchange operation.
        This guarentees that any external synchronisation for these operations are merged allowing for the operations
        to execute in parallel.

        Only applies to operations the current graph. Used as a contextmanager:
        ```
        with pir.merge_exchange():
            ops.host_load(..)
            ops.host_store(..)
        ```

        Note: Operations must be able to be scheduled in any order to be merged. For this reason it is recommended to combine with
            `with pir.in_sequence(False)` to avoid topological constraints that would prevent merging.
    """
    ctx = get_current_context()
    graph = ctx.graph
    ops: Set[int] = set()

    def hook(op: _ir.Op):
        ops.add(op.id)

    handle = graph.register_op_created_hook(hook)
    yield
    graph.remove_op_created_hook(handle)

    ops_created = _ir.transforms.MergeExchange().applyToOps(
        graph._pb_graph, ops)
    for op in ops_created:
        ctx._op_created(op)
