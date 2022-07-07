# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Set
from contextlib import contextmanager
import popart._internal.ir as _ir
from popxl.context import io_tiles, in_sequence, get_current_graph, get_current_context


@contextmanager
def io_tile_exchange(verify_overlap: bool = True):
    """Combine io_tiles, merge_exchange and in_sequence(False).

    Args:
        verify_overlap (bool, optional): Verify only one Operation remains after the context closes.
                                         This is an important requirement for overlapping IO and Compute.
                                         Defaults to True.

    Raises:
        RuntimeError: If more than one Op remains after io_tile_exchange.
    """
    graph = get_current_graph()

    added_ops: Set[int] = set()

    def hook(op: _ir.Op):
        if not isinstance(op, _ir.op.InitOp):
            added_ops.add(op.id)

    handle = graph.register_op_created_hook(hook)

    with io_tiles(), merge_exchange(), in_sequence(False):
        yield

    graph.remove_op_created_hook(handle)

    if verify_overlap:
        # merge_exchange will remove some operations that were added.
        # verification should only be on the ops remaining in the graph.
        graph_ops = set(graph._pb_graph.getOpIds())
        remaining_ops = added_ops.intersection(graph_ops)
        if len(remaining_ops) > 1:
            ops_debug = '\n   '.join(
                graph._pb_graph.getOp(opid).debugName()
                for opid in remaining_ops)
            raise RuntimeError(
                "More than one Op remained after `io_tile_exchange`. "
                "This will prevent overlap with following compute. Remaining Ops:"
                f"\n   {ops_debug}")


@contextmanager
def merge_exchange():
    """Combine RemoteLoad/RemoteStore/HostLoad/HostStore operations into a single MergeExchange operation.
       This guarantees that any external synchronisation for these operations are merged allowing for the operations
       to execute in parallel.

       Only applies to operations on the current graph. Used as a contextmanager:

       .. code-block:: python

           with popxl.merge_exchange():
               ops.host_load(..)
               ops.host_store(..)

       Note: Operations must be able to be scheduled in any order to be merged. For this reason it is recommended to combine with
       `with popxl.in_sequence(False)` to avoid topological constraints that would prevent merging. Related: :py:meth:`io_tile_exchange`.
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
