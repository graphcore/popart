# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple, TypeVar, Union, overload
from typing_extensions import Literal
import inspect
import os
from collections import defaultdict
from functools import wraps
from contextlib import contextmanager

import popart._internal.ir as _ir

if TYPE_CHECKING:
    from popxl.graph import Graph
    from popxl.tensor import Tensor

InSequenceKey = Tuple[_ir.GraphId, _ir.ExecutionContext]


class Context:
    def __init__(self):
        self._reset()
        self._patch_op_listener()

    def _reset(self):
        self._graphs: List['Graph'] = []
        self._ipu_id: int = 0
        self._in_sequence: Optional[bool] = None
        self._previous_ops: DefaultDict[InSequenceKey,
                                        List[int]] = defaultdict(list)
        self._debug_info: Optional[_ir.DebugInfo] = None
        self._debug_context_frame_offset: int = 0
        self._name_scope: List[str] = []
        self._io_tile_set: bool = False
        self._execution_context: _ir.ExecutionContext = _ir.ExecutionContext.Normal

        self._hook_handle: int = 0
        self._op_created_hooks: Dict[int, Callable[[_ir.Op], Any]] = {}

    def _get_op_settings(self, name: str) -> _ir.Settings:
        """Return an internal_ir Settings object using any values specified by a context.
            For example: ipu"""
        pb_g = self.graph._pb_graph
        settings = _ir.Settings(pb_g, "/".join((*self.name_scopes, name)))

        vgid = self._ipu_id
        if vgid is not None:
            settings.vgraphId = _ir.OptionalVGraphId(vgid)

        settings.executionContext = self._execution_context

        if self.io_tile_set:
            settings.tileSet = _ir.TileSet.IO

        if self._debug_info is not None:
            settings.debugInfoId = self._debug_info.getId()

        return settings

    @property
    def ipu_id(self) -> Optional[int]:
        return self._ipu_id

    @property
    def io_tile_set(self) -> bool:
        return self._io_tile_set

    def push_name_scope(self, name: str):
        self._name_scope.append(name)

    def pop_name_scope(self) -> str:
        return self._name_scope.pop()

    @property
    def name_scopes(self):
        return list(self._name_scope)

    def push_graph(self, g: 'Graph'):
        if len(self._graphs) > 0 and self._graphs[0].ir != g.ir:
            raise RuntimeError(
                f"Trying to a create a context manager nested in another context, "
                "when the previous context's Ir is different. "
                f"{self._graphs[0].ir} != {g.ir}")
        self._graphs.append(g)

    def pop_graph(self) -> 'Graph':
        return self._graphs.pop()

    @property
    def graph(self):
        if len(self._graphs) == 0:
            raise RuntimeError(
                "Trying to access a graph, but no graph has been selected. Hint - "
                "try performing the operations in a context manager (for example "
                "`with graph_instance:`) or inside a function that's called by "
                "`popxl.Ir().create_graph()`")
        return self._graphs[-1]

    @property
    def main_graph(self):
        if len(self._graphs) == 0:
            raise RuntimeError(
                "Trying to access the main_graph, but no graph has been selected. Hint - "
                "try performing the operations in a context manager (for example "
                "`with graph_instance:`) or inside a function that's called by "
                "`popxl.Ir().create_graph()`")
        return self._graphs[0].main_graph

    @property
    def in_sequence(self):
        return self._in_sequence

    @in_sequence.setter
    def in_sequence(self, enabled: bool):
        if enabled is None:
            # Must clear all previous Ops when returning
            # to an empty scope.
            self._previous_ops.clear()
        self._in_sequence = enabled

    @in_sequence.deleter
    def in_sequence(self):
        raise AttributeError(
            "Cannot delete in_sequence. Set to 'False' instead.")

    def register_op_created_hook(self, fn: Callable[[_ir.Op], Any]) -> int:
        self._hook_handle += 1
        self._op_created_hooks[self._hook_handle] = fn
        return self._hook_handle

    def remove_op_created_hook(self, handle: int):
        if handle not in self._op_created_hooks.keys():
            raise ValueError(f"Unknown op created hook {handle}")
        self._op_created_hooks.pop(handle)

    def _patch_op_listener(self):
        """Wrap all `createConnectedOp` and `createOp` in the internal library.

            Allowing for the creation of ops to be tracked by popxl.
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
        """
        Call _add_in_sequence_topocons and call all hooks on the operator.

        Used as a callback when an op is created.
        """
        self._add_in_sequence_topocons(op)
        for fn in self._op_created_hooks.values():
            fn(op)

    def _add_in_sequence_topocons(self, op: _ir.Op):
        """Add topocons to ensure operations are executed in sequence.

            If ``op`` is added in an ``in_sequence(True)`` context then
                add a topocon from the Ops in the previous ops.
                Then set the previous ops to only ``op``.
            If ``op`` is added in an ``in_sequence(False)`` context then
                add a topocon from the last op added in a True context.
                Then append ``op`` to previous ops.
            The above behavior results in Ops in a False context to be executed in
            any order. But if the False context is nested inside a True context
            then all Ops within the False context will be treated as a single Op for the purpose
            of the outer True context. For example:

            .. code-block:: python

                with in_sequence(True):
                    OpA()
                    with in_sequence(False)
                        OpB()
                        OpC()
                    OpD()

            OpA will be executed before OpB and OpC. OpD will be executed after OpB and OpC.
            """
        g = op.getGraph()
        ops = g.getOpIds()

        def insert_topocon(before_id: int):
            if before_id != op.id and before_id in ops:
                g.topoCons().insert(g.getOp(before_id), op, False)

        if isinstance(self._in_sequence, bool):
            previous_ops = self._previous_ops[(
                g.id, op.getSettings().executionContext)]
            if self._in_sequence:
                for prev_op in previous_ops:
                    insert_topocon(prev_op)
                previous_ops.clear()
            else:
                prev_ops = previous_ops
                if len(prev_ops) > 0:
                    insert_topocon(prev_ops[0])
                else:
                    prev_ops.append(-1)

            # Mutates value in self._previous_ops as `previous_ops` is same reference
            previous_ops.append(op.id)

        elif self.in_sequence is None:
            # `pass` is converted to None
            pass

        else:
            raise ValueError(
                f"`in_sequence` mode not understood: {self.in_sequence}. "
                "Must be one of `True`, `False`, `'pass'`.")


_CURRENT_CONTEXT = Context()


def get_current_context() -> Context:
    global _CURRENT_CONTEXT
    return _CURRENT_CONTEXT


def get_current_graph() -> 'Graph':
    """
    Get the current graph from the current context.

    The function `gcg()` is available as an alias of `get_current_graph()`.

    Raises:
        RuntimeError:
            If the stack is empty.

    Returns:
        Graph:
            The last graph that was used as a context scope.
    """
    return get_current_context().graph


def get_main_graph() -> 'Graph':
    """
    Get the main graph from the current context.

    The function `gmg()` is available as an alias of `get_main_graph()`.

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
def ipu(ipu: int):
    """Set the IPU for ops created in this context."""
    # PopART internally uses Poplar virtual graphs.
    ctx = get_current_context()
    prev = ctx._ipu_id
    ctx._ipu_id = ipu
    yield ipu
    ctx._ipu_id = prev


@contextmanager
def in_sequence(mode: Union[bool, Literal['pass']] = True):
    """Force ops created in this context to execute in the order that they are created.

    This is achieved by adding topological constraints to the scheduling graph.

    Args:
        mode (Union[bool, Literal['pass']]):
            - ``True``: Ensure each op within the context is executed in a linear schedule
              in the same order as created.
            - ``False``: Do not apply linear schedule constraint. If nested within an outer
              ``in_sequence(True)`` context, the inner context as a whole will be scheduled
              linearly with respect to ops in the outer context. For example:

              .. code-block:: python

                  with in_sequence(True):
                      OpA()
                      with in_sequence(False)
                          OpB()
                          OpC()
                      OpD()

              `OpA` will be executed before `OpB` and `OpC`.

              `OpD` will be executed after `OpB` and `OpC`.

            - ``'pass'``: Do nothing when an op is created. Useful for transforms that want to
              delay adding topological constraints to the graph.

    Raises:
        TypeError: If None is passed to ``mode``.
    """

    # We use `None` to specify an empty scope. It must not be passed here:
    if mode is None:
        raise TypeError(
            "`None` cannot be passed to `in_sequence`. Try `False` instead.")

    if mode == 'pass':
        mode = None

    ctx = get_current_context()
    prev = ctx.in_sequence
    ctx.in_sequence = mode
    yield mode
    ctx.in_sequence = prev  # type: ignore


@contextmanager
def io_tiles():
    """Execute ops created in this context on IO tiles of the current IPU."""
    ctx = get_current_context()
    prev = ctx._io_tile_set
    ctx._io_tile_set = True
    yield
    ctx._io_tile_set = prev


@contextmanager
def name_scope(name: str):
    """Set the name scope for ops created in this context."""
    ctx = get_current_context()
    ctx.push_name_scope(name)
    yield ctx.name_scopes
    ctx.pop_name_scope()


@contextmanager
def debug_context_frame_offset(i: int):
    ctx = get_current_context()
    # Plus 1 to account for frame usage as a decorator
    ctx._debug_context_frame_offset += (i + 1)
    yield ctx._debug_context_frame_offset
    ctx._debug_context_frame_offset -= (i + 1)


@contextmanager
def _execution_context(execution_context: _ir.ExecutionContext):
    ctx = get_current_context()
    prev = ctx._execution_context
    ctx._execution_context = execution_context
    yield execution_context
    ctx._execution_context = prev


def _tensor_ids_from_maybe_tensors(
        ts: Union['Tensor', Iterable[Any]]) -> _ir.ProfileValue:
    from popxl.tensor import Tensor
    if isinstance(ts, Tensor):
        ids = [_ir.ProfileValue(ts.id)]
    else:
        ids = [_ir.ProfileValue(t.id) for t in ts if isinstance(t, Tensor)]
    return _ir.ProfileValue(ids)


def get_source_location(offset: int):
    stack = inspect.getouterframes(inspect.currentframe())
    try:
        debug_frame = stack[offset + 1]
    except IndexError as e:
        raise IndexError(
            f"Incorrect source location offset. Stack {len(stack)}, Offset {offset+1}"
        ) from e
    return _ir.SourceLocation(debug_frame.function,
                              os.path.realpath(debug_frame.filename),
                              debug_frame.lineno)


# overload is required so that `op_debug_context` does not remove typehints
Fn = TypeVar('Fn')


@overload
def op_debug_context(name: str) -> Callable[[Fn], Fn]:
    ...


@overload
def op_debug_context(name: Fn) -> Fn:
    ...


def op_debug_context(name):  # type: ignore
    """
    Specify a new op debug context.

    Commonly used as decorator.

    Typical usage:

    .. code-block:: python

        @op_debug_context
        def add(lhs, rhs):
            ...

    Or, to specify the context name:

    .. code-block:: python

        @op_debug_context("op")
        def my_op(x):
            ...
    """
    _name: str = name  # type: ignore

    def outer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = get_current_context()
            prev = ctx._debug_info

            # TODO: Allow for parent layer DebugInfo to be passed here.
            dc = _ir.DebugContext(
                "/".join((*ctx.name_scopes, _name)),
                get_source_location(ctx._debug_context_frame_offset + 1))

            di = _ir.DebugInfo(dc, "popxl")
            di.setValue("category", "op")
            di.setValue("api", _name)

            inputs = _tensor_ids_from_maybe_tensors((*args, *kwargs.values()))
            di.setValue("inputs", inputs)
            ctx._debug_info = di

            # Run function
            result = func(*args, **kwargs)

            if result is not None:
                try:
                    outputs = _tensor_ids_from_maybe_tensors(result)
                    di.setValue("outputs", outputs)
                except TypeError:
                    pass

            ctx._debug_info = prev
            return result

        return wrapper

    if callable(name):
        _name = name.__name__
        return outer(name)
    return outer
