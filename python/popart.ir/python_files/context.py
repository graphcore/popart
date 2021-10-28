# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Callable, DefaultDict, List, Optional
from collections import defaultdict
from functools import wraps
from contextlib import contextmanager
import popart._internal.ir as _ir

if TYPE_CHECKING:
    from popart.ir.graph import Graph

__all__ = [
    'get_current_graph', 'get_main_graph', 'gcg', 'gmg', 'virtual_graph',
    'pipeline_stage', 'in_sequence'
]


class Context:
    def __init__(self):
        self._reset()
        self._patch_op_listener()

    def _reset(self):
        self._graphs: List['Graph'] = []
        # By default place all ops on ipu 0
        self._virtual_graph_ids: List[int] = [0]
        self._pipeline_stages: List[int] = []
        self._in_sequence: Optional[bool] = None
        self._previous_ops: DefaultDict[_ir.GraphId, List[
            _ir.Op]] = defaultdict(list)

    def _get_op_settings(self, name: str) -> _ir.Settings:
        """Return an internal_ir Settings object using any values specified by a context.
            For example: virtual_graph"""
        pb_g = self.graph._pb_graph
        settings = _ir.Settings(pb_g, name)

        vgid = self.virtual_graph_id
        if vgid is not None:
            settings.vgraphId = _ir.OptionalVGraphId(vgid)

        pstage = self.pipeline_stage
        if pstage is not None:
            settings.pipelineStage = _ir.OptionalPipelineStage(pstage)

        return settings

    def push_virtual_graph_id(self, vgid: int):
        self._virtual_graph_ids.append(vgid)

    def pop_virtual_graph_id(self) -> int:
        return self._virtual_graph_ids.pop()

    @property
    def virtual_graph_id(self) -> Optional[int]:
        return self._virtual_graph_ids[-1] if len(
            self._virtual_graph_ids) > 0 else None

    def push_pipeline_stage(self, stage: int):
        self._pipeline_stages.append(stage)

    def pop_pipeline_stage(self) -> int:
        return self._pipeline_stages.pop()

    @property
    def pipeline_stage(self) -> Optional[int]:
        return self._pipeline_stages[-1] if len(
            self._pipeline_stages) > 0 else None

    def push_graph(self, g: 'Graph'):
        if len(self._graphs) > 0 and self._graphs[0].ir() != g.ir():
            raise RuntimeError(
                f"Trying to a create a context manager nested in another context, "
                "when the previous context's Ir is different. "
                f"{self._graphs[0].ir()} != {g.ir()}")
        self._graphs.append(g)

    def pop_graph(self) -> 'Graph':
        return self._graphs.pop()

    @property
    def graph(self):
        if len(self._graphs) == 0:
            raise RuntimeError(
                "Trying to access a graph, but no graph has been selected. Hint - "
                "try performing the operations in a context manager (e.g. "
                "`with graph_instance:`) or inside of a function that's called by "
                "`popart.ir.Ir().create_graph()`")
        return self._graphs[-1]

    @property
    def main_graph(self):
        if len(self._graphs) == 0:
            raise RuntimeError(
                "Trying to access the main_graph, but no graph has been selected. Hint - "
                "try performing the operations in a context manager (e.g. "
                "`with graph_instance:`) or inside of a function that's called by "
                "`popart.ir.Ir().create_graph()`")
        return self._graphs[0].get_main_graph()

    @property
    def in_sequence(self):
        return self._in_sequence

    @in_sequence.setter
    def in_sequence(self, enabled: bool):
        if enabled is None:
            # Must clear all previous Ops when returning
            # to an empty scope.
            self._previous_ops.clear()
        if self._in_sequence is None and not enabled:
            # False scope inside an empty scope should have no
            # effect.
            return
        self._in_sequence = enabled

    @in_sequence.deleter
    def in_sequence(self):
        raise AttributeError(
            "Cannot delete in_sequence. Set to 'False' instead.")

    def _patch_op_listener(self):
        """Wraps all `createConnectedOp` and `createOp` in the internal library.
            Allowing for the creation of ops to be tracked by popart.ir.
            Currently this is used for adding topological constraints to linearize graphs."""

        def _register_listener(name: str, fn: Callable):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                op = fn(*args, **kwargs)
                self._op_created(op)
                return op

            setattr(_ir.Graph, name, wrapper)

        for method in dir(_ir.Graph):
            if "createConnectedOp" in method or "createOp" in method:
                _register_listener(method, getattr(_ir.Graph, method))

    def _op_created(self, op: _ir.Op):
        """Callback for when an op is created."""
        self._add_in_sequence_topocons(op)

    def _add_in_sequence_topocons(self, op):
        """Adds topocons to ensure operations are executed in sequence.
            If #op is added in an `in_sequence(True)` context then
                add a topocon from the Ops in the previous ops.
                Then set the previous ops to only #op.
            If #op is added in an `in_sequence(False)` context then
                add a topocon from the last op added in a True context.
                Then append #op to previous ops.
            The above behaviour results in Ops in a False context to be executed in
            any order. But if the False context is nested inside a True context
            then all Ops within the False context will be treated as a single Op for the purpose
            of the outer True context. For example:
            ```
                with in_sequence(True):
                    OpA()
                    with in_sequence(False)
                        OpB()
                        OpC()
                    OpD()
            ```
            OpA will be executed before OpB and OpC. OpD will be executed after OpB and OpC.
            """
        g = op.getGraph()
        if self._in_sequence is not None:
            if self._in_sequence:
                for prev_op in self._previous_ops[g.id]:
                    g.topoCons().insert(prev_op, op, False)
                self._previous_ops[g.id].clear()
            else:
                prev_ops = self._previous_ops[g.id]
                if len(prev_ops) > 0:
                    g.topoCons().insert(prev_ops[0], op, False)
            self._previous_ops[g.id].append(op)


_CURRENT_CONTEXT = Context()


def get_current_context() -> Context:
    global _CURRENT_CONTEXT
    return _CURRENT_CONTEXT


def get_current_graph() -> 'Graph':
    """
    Get the current graph from the current context.

    Raises:
        RuntimeError:
            If the stack is empty.

    Returns:
        Graph:
            The graph at the top of the global graph stack.
    """
    return get_current_context().graph


def get_main_graph() -> 'Graph':
    """
    Get the main graph from the current context.

    Raises:
        RuntimeError:
            If the stack is empty.

    Returns:
        Graph:
            The main graph.
    """
    return get_current_context().main_graph


# Alias for get_current_graph() and get_main_graph()
gcg = get_current_graph
gmg = get_main_graph


@contextmanager
def virtual_graph(vgid: int):
    """Set the virtual graph id on Ops created in this context."""
    ctx = get_current_context()
    ctx.push_virtual_graph_id(vgid)
    yield vgid
    ctx.pop_virtual_graph_id()


@contextmanager
def pipeline_stage(stage: int):
    """Set the pipeline stage on Ops created in this context."""
    ctx = get_current_context()
    ctx.push_pipeline_stage(stage)
    yield stage
    ctx.pop_pipeline_stage()


@contextmanager
def in_sequence(enabled: bool = True):
    """Force Ops created in this context to executing in the order that they are created."""

    # We use `None` to specify an empty scope. It must not be passed here:
    if enabled is None:
        raise TypeError(
            "None cannot be passed to 'in_sequence'. Try False instead.")

    ctx = get_current_context()
    prev = ctx.in_sequence
    ctx.in_sequence = enabled
    yield enabled
    ctx.in_sequence = prev
