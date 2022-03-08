# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Auto diff transformation."""
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import popart._internal.ir as _ir
from popxl.graph import Graph
from popxl.tensor import Tensor
from popxl.ops.call import CallSiteInfo


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
        return f"({self.connection_type}, {self.fwd_tensor})"

    @property
    def fwd_tensor(self) -> Tensor:
        return Tensor._from_pb_tensor(
            self._fwd_graph.getTensor(self._pb_ec.fwdId))

    @property
    def connection_type(self):
        return ExpectedConnectionType._from_pb(self._pb_ec.type)


class GradGraphInfo:
    """
    The result of differentiating a graph.
    `graph` is the computational graph for computing the gradient
    `expected_inputs` are Tensors from the forward_graph
        that are required as inputs to the grad `graph`
    `expected_outputs` are Tensors from the forward_graph that have gradients as
        outputs of the grad `graph`.
    """

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
        """The graph that was autodiffed to produce the grad graph"""
        return Graph._from_pb(self._fwd_graph)

    @property
    def expected_inputs(self) -> Tuple[ExpectedConnection, ...]:
        """
        Get information of all inputs of the gradient graph.
        Inputs are tensors or their gradients from the `forward_graph`
        that are required as inputs to the gradient graph.
        """
        return tuple(self._expected_inputs)

    @property
    def expected_outputs(self) -> Tuple[ExpectedConnection, ...]:
        """
        Get information of all outputs of the gradient graph.
        Outputs are tensors from the `forward_graph` that have gradients as
        outputs of the gradient graph.
        """
        return tuple(self._expected_outputs)

    def __repr__(self) -> str:
        """Return a string representation for this GradGraphInfo object."""
        result = "GradGraphInfo[\n"
        result += f"  graph={self.graph.id}\n"
        result += "  expected_inputs=[\n"
        result += "".join([f"    {repr(ec)}\n" for ec in self.expected_inputs])
        result += "  ]\n"
        result += "  expected_outputs=[\n"
        result += "".join(
            [f"    {repr(ec)}\n" for ec in self.expected_outputs])
        result += "  ]\n"
        result += "]"
        return result

    @property
    def inputs(self) -> Tuple[Tensor, ...]:
        """Get a tuple of all inputs of the gradient graph."""
        return tuple(map(lambda ec: ec.fwd_tensor, self._expected_inputs))

    @property
    def outputs(self) -> Tuple[Tensor, ...]:
        """Get a tuple of all outputs of the gradient graph."""
        return tuple(map(lambda ec: ec.fwd_tensor, self._expected_outputs))

    def inputs_dict(self, fwd_call_info: CallSiteInfo) -> Dict[Tensor, Tensor]:
        """
        Returns a gradient graph `inputs_dict` inputs from a forward's graph call site.

        The output dictionary maps inputs of the gradient graph to inputs in the parent graph.

        Example:

        .. code-block:: python

            # `module`: subgraph module, `x` parent graph inputs, `x_dash` grad graph parent input
            graph = ir.create_graph(module, x, out_features=16) # Forwards graph
            call_info = ops.call_with_info(
                graph, x, inputs_dict={module.W: W, module.b: b})

            grads_graph = popxl.transforms.autodiff(graph)
            activations = ss_bwd_info.inputs_dict(call_info)
            grads_call_info = ops.call_with_info(
                grads_graph, x_dash, inputs_dict=activations)

        Args:
            fwd_call_info (CallSiteInfo):
                Callsite info of a call to the forwards graph that was auto-differentiated. This can be accessed by
                using `ops.call_with_info()`

        Returns: `Dict[Tensor, Tensor]`
            from: a Tensor in the gradient Graph
            to: an input or output parent graph tensor at a callsite of the corresponding forward Graph.
        """
        if fwd_call_info.called_graph != self.forward_graph:
            raise TypeError(
                "The called graph does not match the graph that was auto-differentiated."
            )

        return {
            Tensor._from_pb_tensor(self.graph._pb_graph.getInputTensor(idx)):
            fwd_call_info.graph_to_parent(act.fwd_tensor)
            for idx, act in enumerate(self.expected_inputs)
            if act.connection_type == ExpectedConnectionType.Fwd
        }

    def fwd_graph_ins_to_grad_parent_outs(
            self, grad_call_info: CallSiteInfo) -> Dict[Tensor, Tensor]:
        """
        Returns a mapping between a forward graph inputs to a gradient's parent graph outputs.

        Example:

        .. code-block:: python

            # `module`: subgraph module, `x` parent graph inputs, `x_dash` grad parent graph input
            graph = ir.create_graph(module, x, out_features=16) # Forwards graph
            call_info = ops.call_with_info(
                graph, x, inputs_dict={module.W: W, module.b: b})

            grads_graph = popxl.transforms.autodiff(graph)
            activations = ss_bwd_info.inputs_dict(call_info)
            grads_call_info = ops.call_with_info(
                grads_graph, x_dash, inputs_dict=activations)

            # Obtain a mapping between subgraph tensor that corresponds to `x`, `W` and `b` and the corresponding parent grad Tensors outputs
            grad_tensor_map = grads_graph.fwd_graph_ins_to_grad_parent_outs(grads_call_info)
            assert all(t in graph for t in grad_tensor_map.keys())
            assert all(t in main for t in grad_tensor_map.values())
            assert [t.id for t in grad_tensor_map.keys()] == [
                'Module_subgraph(0)/x', 'Module_subgraph(0)/W', 'Module_subgraph(0)/b']
            assert [t.id for t in grad_tensor_map.values()] == [
                'Gradient___x', 'Gradient___W', 'Gradient___b']

        Args:
            grad_call_info (CallSiteInfo):
                Callsite info of a call to the graph that was auto-differentiated. This can be accessed by
                using `ops.call_with_info()`

        Returns: `Dict[Tensor, Tensor]`
            from: a Tensor in the forwards graph
            to: corresponding Tensor in the gradient graph
        """
        return {
            t: grad_call_info.graph_to_parent(self.graph.outputs[idx])
            for idx, t in enumerate(self.outputs)
        }

    def fwd_parent_ins_to_grad_parent_outs(
            self, fwd_call_info: CallSiteInfo,
            grad_call_info: CallSiteInfo) -> Dict[Tensor, Tensor]:
        """
        Returns a mapping between forward's parent graph inputs to a gradient's parent graph outputs.

        Example:

        .. code-block:: python

            # `module`: subgraph module, `x` graph inputs, `x_dash` grad graph input
            graph = ir.create_graph(module, x, out_features=16) # Forwards graph
            call_info = ops.call_with_info(
                graph, x, inputs_dict={module.W: W, module.b: b})

            grads_graph = popxl.transforms.autodiff(graph)
            activations = ss_bwd_info.inputs_dict(call_info)
            grads_call_info = ops.call_with_info(
                grads_graph, x_dash, inputs_dict=activations)

            # Obtain a mapping input tensors `x`, `W` and `b` and the corresponding grad Tensors
            grad_tensor_map = grads_graph.fwd_parent_ins_to_grad_parent_outs(call_info, grads_call_info)
            assert [t.id for t in grad_tensor_map.keys()] == ['x', 'W', 'b']

        Args:
            fwd_call_info (CallSiteInfo):
                Callsite info of a call to the forwards graph that was auto-differentiated. This can be accessed by
                using `ops.call_with_info()`
            grad_call_info (CallSiteInfo):
                Callsite info of a call to the associated gradient graph. This can be accessed by
                using `ops.call_with_info()`

        Returns: `Dict[Tensor, Tensor]`
            from: a input Tensor for the forwards graph
            to: corresponding Tensor in the gradient graph
        """
        return {
            fwd_call_info.graph_to_parent(t):
            grad_call_info.graph_to_parent(self.graph.outputs[idx])
            for idx, t in enumerate(self.outputs)
        }


def autodiff(graph: Graph,
             grads_provided: Optional[Iterable[Tensor]] = None,
             grads_required: Optional[Iterable[Tensor]] = None,
             called_graphs_grad_info: Optional[
                 Mapping[Graph, GradGraphInfo]] = None,
             return_all_grad_graphs: bool = False):
    """
    Differentiate a Graph.

    The graph will be differentiated using the chain rule starting from `grads_provided`.
    The outputs of the returned graph will be the gradient of the Tensors in `grads_required`.
    By default `grad_provided` will be all of the outputs of the forward graph and `grads_required` will
    be all of the inputs to the forward graph.

    Any Tensors in the forward graph that are needed to compute the gradients will be added as outputs
    to the forward graph (if not already an input/output).

    The returned `GradGraphInfo` contains the gradient graph and information
    regarding the graph inputs (`expected_inputs`) and (`graph outputs`) of
    the gradient graph. These are lists of tuples where the first element is
    either `ExpectedConnectionType.Fwd` or `ExpectedConnectionType.FwdGrad`
    meaning the input/output is associated with a tensor in the forward
    graph, or the gradient of a tensor in the forward graph, respectively.
    The second element is a tensor of the forward graph itself. These
    tensors are guaranteed to be either inputs or outputs of the forward
    graph.

    The `expected_inputs` list that describes the gradient graph's inputs
    is guaranteed to start with `ExpectedConnectionType.FwdGrad` entries
    that exactly match order of the `grads_provided` parameter.

    The `expected_outputs` list that describes the gradient graph's outputs
    is guaranteed to comprise only `ExpectedConnectionType.FwdGrad` entries
    and has entries that exactly match the size and order of the
    `grads_required` parameter.

    Any graphs called in the forward graph will recursively have `autodiff` called on it. Arg `called_graphs_grad_info` can be
    used to specify the result of `autodiff` on a called graph that has already been differentiated.
    By default GradGraphInfo will only be returned for the provided forward graph. Arg `return_all_grad_graphs` can be set to `True` to return
    info on all graphs that `autodiff` as executed on as a result of this transformation.

    Args:
        graph (popxl.Graph): Graph to autodiff
        grads_provided (Optional[Iterable[popxl.Tensor]], optional): Defaults to all outputs of the provided graph.
        grads_required (Optional[Iterable[popxl.Tensor]], optional); Defaults to all inputs of the provided graph.
        called_graphs_grad_info (Optional[Mapping[popxl.Graph, GradGraphInfo]], optional): Defaults to None.
        return_all_grad_graphs (bool, optional): Defaults to False.

    Returns:
        grad_info: GradGraphInfo
    """

    grads_provided = graph.outputs if grads_provided is None else grads_provided
    grads_required = graph.inputs if grads_required is None else grads_required
    called_graphs_grad_info = {} if called_graphs_grad_info is None else called_graphs_grad_info

    _pb_ir = graph.ir._pb_ir
    transform = _ir.transforms.Autodiff()

    _pb_result = transform.apply(
        _pb_ir, _ir.GraphId(graph.name), [t.id for t in grads_provided],
        _ir.OptionalTensors([t.id for t in grads_required]),
        {k: v._pb_bwd_info
         for k, v in called_graphs_grad_info.items()})

    result: Mapping[Graph, GradGraphInfo] = {}
    for k, v in _pb_result.items():
        _graph = Graph._from_pb(_pb_ir.getGraph(k))
        result[_graph] = GradGraphInfo._from_pb(_pb_ir, _graph._pb_graph, v)

    if return_all_grad_graphs:
        return result

    return result[graph]
