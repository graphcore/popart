# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Auto diff transformation."""
from enum import Enum
from typing import Iterable, List, Mapping, Optional, Tuple

import popart._internal.ir as _ir
from popart.ir.graph import Graph
from popart.ir.tensor import Tensor
from popart.ir.ops.call import CallInfo

__all__ = [
    'autodiff', 'get_expected_forward_inputs_from_call',
    'ExpectedConnectionType'
]


class ExpectedConnectionType(Enum):
    Fwd = "Fwd"
    FwdGrad = "FwdGrad"

    @classmethod
    def _from_pb(cls, type_: _ir.ExpectedConnectionType):
        if type_ == _ir.ExpectedConnectionType.Fwd:
            return ExpectedConnectionType.Fwd
        return ExpectedConnectionType.FwdGrad

    def _to_pb(self):
        if self == ExpectedConnectionType.Fwd:
            return _ir.ExpectedConnectionType.Fwd
        return _ir.ExpectedConnectionType.FwdGrad


class ExpectedConnection:
    def __init__(self) -> None:
        self._fwd_graph: _ir.Graph
        self._pb_ec: _ir.ExpectedConnection
        raise TypeError(
            "ExpectedConnection should not be constructed directly. Use ExpectedConnection._from_pb instead."
        )

    @classmethod
    def _from_pb(cls, fwd_graph: _ir.Graph,
                 pb_ec: _ir.ExpectedConnection) -> "ExpectedConnection":
        self = super().__new__(cls)
        self._fwd_graph = fwd_graph
        self._pb_ec = pb_ec
        return self

    def __repr__(self) -> str:
        return f"({self.connection_type}, {self.tensor})"

    @property
    def tensor(self) -> Tensor:
        return Tensor._from_pb_tensor(
            self._fwd_graph.getTensor(self._pb_ec.fwdId))

    @property
    def connection_type(self):
        return ExpectedConnectionType._from_pb(self._pb_ec.type)


class GradGraphInfo:
    """The result of differentiating a graph.
        `graph` is the computational graph for computing the gradient
        `expected_inputs` are Tensors from the forward_graph
            that are required as inputs to the grad `graph`
        `expected_outputs` are Tensors from the forward_graph that have gradients as
            outputs of the grad `graph`."""

    def __init__(self) -> None:
        self._ir: _ir.Ir
        self._fwd_graph: _ir.Graph
        self._expected_inputs: List[ExpectedConnection]
        self._expected_outputs: List[ExpectedConnection]
        self._pb_bwd_info: _ir.BwdGraphInfo
        raise TypeError(
            "GradGraphInfo should not be constructed directly. Use GradGraphInfo._from_pb instead."
        )

    @classmethod
    def _from_pb(cls, ir: _ir.Ir, fwd_graph: _ir.Graph,
                 _pb_bwd_info: _ir.BwdGraphInfo) -> "GradGraphInfo":
        self = super().__new__(cls)
        self._ir = ir
        self._fwd_graph = fwd_graph
        self._expected_inputs = [
            ExpectedConnection._from_pb(fwd_graph, e)
            for e in _pb_bwd_info.expectedInputs
        ]
        self._expected_outputs = [
            ExpectedConnection._from_pb(fwd_graph, e)
            for e in _pb_bwd_info.expectedOutputs
        ]
        self._pb_bwd_info = _pb_bwd_info
        return self

    @property
    def graph(self):
        """The computational graph for computing the gradient"""
        _id = self._pb_bwd_info.bwdGraphId
        return Graph._from_pb(self._ir.getGraph(_id))

    @property
    def forward_graph(self):
        return Graph._from_pb(self._fwd_graph)

    @property
    def expected_inputs(self) -> Tuple[ExpectedConnection, ...]:
        """Tensors (or their gradients) from the forward_graph
            that are required as inputs to the gradient graph"""
        return tuple(self._expected_inputs)

    @property
    def expected_outputs(self) -> Tuple[ExpectedConnection, ...]:
        """Tensors from the forward_graph that have gradients as
            outputs of the gradient graph"""
        return tuple(self._expected_outputs)

    def get_input_tensors(self) -> Tuple[Tensor, ...]:
        return tuple(map(lambda ec: ec.tensor, self._expected_inputs))

    def get_output_tensors(self) -> Tuple[Tensor, ...]:
        return tuple(map(lambda ec: ec.tensor, self._expected_outputs))


def autodiff(
        graph: Graph,
        gradsProvided: Optional[Iterable[Tensor]] = None,
        gradsRequired: Optional[Iterable[Tensor]] = None,
        calledGraphsGradInfo: Optional[Mapping[Graph, GradGraphInfo]] = None,
        returnAllGradGraphs: bool = False):
    """Differentiate a Graph.

        The graph will be differentiated using the chain rule starting from `gradsProvided`.
        The outputs of the returned graph will be the gradient of the Tensors in `gradsRequired`.
        By default `gradProvided` will be all of the outputs of the forward graph and `gradsRequired` will
        be all of the inputs to the forward graph.

        Any Tensors in the forward graph that are needed to compute the gradients will be added as outputs
        to the forward graph (if not already an input/output).

        The returned `GradGraphInfo` contains the gradient graph and information regarding all required inputs
        to the gradient graph. This can include tensors which are outputs of the forward graph `ExpectedConnectionType.Fwd`,
        or a gradient of an output of the forwards graph `ExpectedConnectionType.FwdGrad`.

        Any graphs called in the forward graph will recursively have `autodiff` called on it. Arg `calledGraphsGradInfo` can be
        used to specify the result of `autodiff` on a called graph that has already been differentiated.
        By default GradGraphInfo will only be returned for the provided forward graph. Arg `returnAllGradGraphs` can be set to `True` to return
        info on all graphs that `autodiff` as executed on as a result of this transformation."""

    gradsProvided = graph.get_output_tensors(
    ) if gradsProvided is None else gradsProvided
    gradsRequired = graph.get_input_tensors(
    ) if gradsRequired is None else gradsRequired
    calledGraphsGradInfo = {} if calledGraphsGradInfo is None else calledGraphsGradInfo

    _pb_ir = graph.ir()._pb_ir
    transform = _ir.transforms.Autodiff()

    _pb_result = transform.apply(
        _pb_ir, _ir.GraphId(graph.name), [t.id for t in gradsProvided],
        _ir.OptionalTensors([t.id for t in gradsRequired]),
        {k: v._pb_bwd_info
         for k, v in calledGraphsGradInfo.items()})

    result: Mapping[str, GradGraphInfo] = {}
    for k, v in _pb_result.items():
        result[k.str()] = GradGraphInfo._from_pb(_pb_ir, graph._pb_graph, v)

    if returnAllGradGraphs:
        return result

    return result[graph.name]


def get_expected_forward_inputs_from_call(
        call_info: CallInfo,
        grad_info: GradGraphInfo) -> Mapping[Tensor, Tensor]:
    """Utility function for constructing inputs to calling a grad graph.

      Args:
        call_info: `popart.ir.ops.call.CallInfo`
            Callsite info of a call to the graph that was auto-differentiated. This can be accessed by
            using `ops.call_with_info()`
        grad_info: `GradGraphInfo`
            Output of autodiff on a graph.

      Returns: `Mapping[Tensor, Tensor]`
        from: a Tensor in the gradient Graph
        to: an input or output tensor at a callsite of the corresponding forward Graph.
    """
    if call_info.called_graph != grad_info.forward_graph:
        raise TypeError(
            "The called graph does not match the graph that was auto-differentiated."
        )

    return {
        Tensor._from_pb_tensor(grad_info.graph._pb_graph.getInputTensor(idx)):
        call_info.subgraph_to_op_tensor(act.tensor)
        for idx, act in enumerate(grad_info.expected_inputs)
        if act.connection_type == ExpectedConnectionType.Fwd
    }
