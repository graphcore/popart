# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Set, Dict

import popart._internal.ir as _ir

from popxl.tensor import Tensor, Variable, Constant
from popxl.context import get_current_context

if TYPE_CHECKING:
    from popxl.ir import Ir


class Graph:
    def __init__(self):
        """
        Construct a ``Graph`` object.

        The ``Graph`` class represents graphs in the PopART IR.

        This class cannot be instantiated using the constructor. ``Graph``
        instances are created by the ``Ir`` class. During construction, each ``Ir``
        constructs a default ``main`` graph. More ``Graph`` instances can be made
        using the ``create_graph()`` method of the ``Ir`` class.

        Raises:
            TypeError:
                If an attempt is made to create an object of this class.
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
        """
        Construct `Graph` instances.

        Used as a factory method.

        This method explicitly creates a new `popxl.Graph`.
        `Graph._from_pb` should be used if a `popxl.Graph` may
        have already been constructed for the `pb_graph`

        Args:
            pb_graph (_ir.Graph):
                An instance of the low-level pybind11 `Graph`.

        Returns:
            Graph
        """
        from popxl.ir import Ir
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
        """Get or construct `Graph` instances.

        Args:
            pb_graph (_ir.Graph):
                An instance of the low-level pybind11 `Graph`.

        Returns:
            Graph
        """
        from popxl.ir import Ir
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

    @property
    def ir(self) -> 'Ir':
        return self._ir

    @property
    def main_graph(self) -> 'Graph':
        return self.ir.main_graph

    @property
    def inputs(self) -> Tuple[Tensor, ...]:
        """
        Get the input tensors of the graph.

        Returns:
            Dict[Tensor, ...]: A tuple of all the input tensors.
        """
        _pb_ins = self._pb_graph.getInputIds()
        return tuple(
            Tensor._from_pb_tensor(self._pb_graph.getTensor(o))
            for o in _pb_ins)

    @property
    def inputs_by_name(self) -> Dict[str, Tensor]:
        """
        Get the input tensors of the graph as a dictionary.

        Returns:
            Dict[str, Tensor]: A dict that maps tensor name to tensor
        """
        return {t.name: t for t in self.inputs}

    @property
    def outputs(self) -> Tuple[Tensor, ...]:
        """
        Get the output tensors of the graph.

        Returns:
            Tuple[Tensor, ...]: A tuple of all the output tensors.
        """
        _pb_outs = self._pb_graph.getOutputIds()
        return tuple(
            Tensor._from_pb_tensor(self._pb_graph.getTensor(o))
            for o in _pb_outs)

    def _create_tensor_id(self, name: Optional[str] = None) -> str:
        """Generate a unique tensor ID.

        If the name already exists in the graph it will be modified
        to make it unique in the graph.
        The graph scope will then be added to construct a globally unique ID.

        Args:
            name (Optional[str]):
                A name which will have an ID appended to make it unique.
                Defaults to `"t"`.

        Returns:
            str:
                The unique id of the tensor.
        """
        name = name if name else 't'
        name = "/".join((*get_current_context().name_scopes, name))
        _id = _ir.addScope(self._pb_graph, name)
        if _id in self._pb_graph:
            _id = self._pb_graph.getIr().createIntermediateTensorId(_id)
        return _id

    def __contains__(self, value: Tensor) -> bool:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Value must be of type popxl.Tensor. Value: {value}. Type: {type(value)}"
            )
        return value.id in self._pb_graph

    def __eq__(self, value: 'Graph') -> bool:
        """Graph equality, based on graph and Ir `id`."""
        if not isinstance(value, Graph):
            raise TypeError(
                f"Value must be of type popxl.Graph. Value: {value}. Type: {type(value)}"
            )
        return self.id == value.id and self.ir == value.ir

    def __hash__(self):
        """Hashes the Graph, based on graph and Ir `id`."""
        return hash((self.id, self.ir))

    def __repr__(self) -> str:
        return f"Graph[id={self.id} name={self.name}]"

    def get_tensor(self, tensor_id: str) -> Tensor:
        """Get tensor using string identifier `tensor_id`."""
        return Tensor._from_pb_tensor(self._pb_graph.getTensor(tensor_id))

    @property
    def tensors(self) -> Tuple[Tensor, ...]:
        """Return all tensors in the graph."""
        return tuple(
            Tensor._from_pb_tensor(t) for t in self._pb_graph.getTensors())

    @property
    def variables(self) -> Tuple[Variable, ...]:
        """Return all variable tensors in the graph."""
        return tuple(
            Variable._from_pb_tensor(t)
            for t in self._pb_graph.getTensorsOfType(_ir.TensorType.Variable))

    @property
    def constants(self) -> Tuple[Constant, ...]:
        """Return all constant tensors in the graph."""
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
        """Register a function to be called after an op is created in the graph.

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
        """Remove an op created hook. `handle` should be the result of calling `Graph.register_op_created_hook`.

        Args:
            handle (int): Handle to an op-created hook.
        """
        get_current_context().remove_op_created_hook(handle)
