# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents graphs in the PopART IR."""
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Set

import popart._internal.ir as _ir

from popart.ir.tensor import Tensor, Variable, Constant
from popart.ir.context import get_current_context

if TYPE_CHECKING:
    from popart.ir.ir import Ir

__all__ = ['Graph']


class Graph:
    def __init__(self):
        """The `Graph` class represents graphs in the PopART IR.

        This class cannot be instantiated using the constructor. `Graph`
        instances are created by the `Ir` class. During construction, each `Ir`
        constructs a default `main` graph. More `Graph` instances can be made
        using the `create_graph()` method of the `Ir` class.

        Raises:
            TypeError:
                The `Graph` class cannot be instantiated.
        """
        # The following attributes and their types are declared here for the
        # sake of Python language servers.
        self._pb_graph: _ir.Graph
        self._by_ref_inputs: Set[Tensor]
        # Back reference to Ir is required to avoid garbage collection of Ir.
        self._ir: 'Ir'

        raise TypeError(f"Cannot create {self.__module__}.Graph instances "
                        "using the constructor.")

    @classmethod
    def _create_from_pb(cls, pb_graph: '_ir.Graph') -> 'Graph':
        """Factory method to construct `Graph` instances.
            This method explicitly creates a new `popart.ir.Graph`.
            `Graph._from_pb` should be used if a `popart.ir.Graph` may
            have already been constructed for the `pb_graph`

        Args:
            pb_graph (_ir.Graph):
                An instance of the low-level pybind11 `Graph`.

        Returns:
            Graph
        """
        from popart.ir.ir import Ir
        ir = Ir._from_pb(pb_graph.getIr())
        self: 'Graph' = super().__new__(cls)
        self._pb_graph = pb_graph
        self._by_ref_inputs = set()
        self._ir = ir

        ir._graph_cache[self.id] = self
        return self

    @classmethod
    def _from_pb(
            cls,
            pb_graph: '_ir.Graph',
    ) -> 'Graph':
        """Method to get or construct `Graph` instances.

        Args:
            pb_graph (_ir.Graph):
                An instance of the low-level pybind11 `Graph`.

        Returns:
            Graph
        """
        from popart.ir.ir import Ir
        ir = Ir._from_pb(pb_graph.getIr())
        _id = pb_graph.id.str()
        if _id in ir._graph_cache:
            return ir._graph_cache[_id]

        return cls._create_from_pb(pb_graph)

    @property
    def name(self) -> str:
        return self._pb_graph.getScope().str()

    @property
    def id(self) -> str:
        return str(self._pb_graph.id.str())

    def ir(self) -> 'Ir':
        return self._ir

    def get_main_graph(self) -> 'Graph':
        return self.ir().main_graph()

    def get_input_tensors(self) -> Tuple[Tensor, ...]:
        """Get the input tensors to the graph.

        Returns:
            Tuple[Tensor, ...]: A tuple of all the input tensors.
        """
        _pb_ins = self._pb_graph.getInputIds()
        return tuple(
            Tensor._from_pb_tensor(self._pb_graph.getTensor(o))
            for o in _pb_ins)

    def get_output_tensors(self) -> Tuple[Tensor, ...]:
        """Get the output tensors from the graph.

        Returns:
            Tuple[Tensor, ...]: A tuple of all the output tensors.
        """
        _pb_outs = self._pb_graph.getOutputIds()
        return tuple(
            Tensor._from_pb_tensor(self._pb_graph.getTensor(o))
            for o in _pb_outs)

    def _create_tensor_id(self, name: Optional[str] = None) -> str:
        """Generate a unique tensor id.

        If the name already exists in the graph it will be modified
        to be made unique in the graph.
        The graph scope will then be added to construct the global unique id.

        Args:
            name (Optional[str]):
                A name which will be appended with an id to make unique.
                Defaults to `t`

        Returns:
            str:
                The unique id of the tensor.
        """
        name = name if name else 't'
        _id = _ir.addScope(self._pb_graph, name)
        if _id in self._pb_graph:
            _id = self._pb_graph.getIr().createIntermediateTensorId(_id)
        return _id

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, Tensor):
            return value.id in self._pb_graph
        return False

    def __eq__(self, other: Any) -> bool:
        """Graph equality, based on graph and Ir `id`"""
        return isinstance(
            other, Graph) and self.id == other.id and self.ir() == other.ir()

    def __hash__(self):
        """Hashes the Graph, based on graph and Ir `id`"""
        return hash((self.id, self.ir()))

    def __repr__(self) -> str:
        return f"Graph[id={self.id} name={self.name}]"

    def get_tensor(self, tensor_id: str) -> Tensor:
        return Tensor._from_pb_tensor(self._pb_graph.getTensor(tensor_id))

    def get_tensors(self) -> Tuple[Tensor, ...]:
        """Return all Tensors in the Graph"""
        return tuple(
            Tensor._from_pb_tensor(t) for t in self._pb_graph.getTensors())

    def get_variables(self) -> Tuple[Variable, ...]:
        """Return all Variable Tensors in the Graph"""
        return tuple(
            Variable._from_pb_tensor(t)
            for t in self._pb_graph.getTensorsOfType(_ir.TensorType.Variable))

    def get_constants(self) -> Tuple[Constant, ...]:
        """Return all Constant Tensors in the Graph"""
        return tuple(
            Constant._from_pb_tensor(t)
            for t in self._pb_graph.getTensorsOfType(_ir.TensorType.Const))

    def __enter__(self):
        get_current_context().push_graph(self)
        return self

    def __exit__(self, *exc):
        get_current_context().pop_graph()
        return False

    def register_op_created_hook(self, fn: Callable[[_ir.Op], Any]):
        """Register a function to be called after an Op is created in the graph.

        Args:
            fn (Callable[[_ir.Op], Any]): Function to be called.

        Returns:
            int: Hook handle. Can be passed to `Graph.remove_op_created_hook` to remove the hook.
        """

        @wraps(fn)
        def hook(op: _ir.Op):
            if Graph._from_pb(op.getGraph()) == self:
                fn(op)

        return get_current_context().register_op_created_hook(hook)

    def remove_op_created_hook(self, handle: int):
        """Remove an Op created hook. `handle` should be the result of calling `Graph.register_op_created_hook`.

        Args:
            handle (int): handle to an Op created hook.
        """
        get_current_context().remove_op_created_hook(handle)
