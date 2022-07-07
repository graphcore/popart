# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Automatic differentiation (autodiff) transform."""
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import popart._internal.ir as _ir
from popxl.graph import Graph
from popxl.tensor import Tensor
from popxl.ops.call import CallSiteInfo


class ExpectedConnectionType(Enum):
    """
    Expected connection type of gradient graph inputs and outputs.

    The expected inputs and expected outputs of the gradient graph are
    associated either with tensors in the forward graph or with gradients of
    tensors in the forward graph.
    """
    #: Expected input/output is a tensor of the forward graph.
    Fwd = "Fwd"

    #: Expected input/output is the gradient of a tensor from the forward graph.
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
    """
    Create connection between tensors and gradient graph inputs or outputs.

    :py:class:`GradGraphInfo` contains the expected inputs and outputs for the
    gradient graph. The expected inputs and outputs are associated either with
    tensors in the forward graph or with gradients of tensors in the forward
    graph.

    This class should not be constructed directly and it raises a ``TypeError``
    exception if it is constructed directly.
    """

    def __init__(self) -> None:
        """Construct the ``ExpectedConnection`` class."""
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
        """Get the tensor this connection applies to."""
        return Tensor._from_pb_tensor(
            self._fwd_graph.getTensor(self._pb_ec.fwdId))

    @property
    def connection_type(self):
        """Get the type of the connection."""
        return ExpectedConnectionType._from_pb(self._pb_ec.type)


class GradGraphInfo:
    """
    Create the result of the ``autodiff`` transform.

    * ``fwd_graph`` is the forward graph.
    * ``expected_inputs`` are tensors from the forward graph that are required as
      inputs to the gradient graph.
    * ``expected_outputs`` are tensors from the forward graph that have gradients
      as outputs of the gradient graph.

    This class should not be constructed directly and it raises a ``TypeError``
    exception if it is constructed directly.
    """

    def __init__(self) -> None:
        """Construct for the ``GradGraphInfo`` class."""
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
        """Get the gradient graph."""
        _id = self._pb_bwd_info.bwdGraphId
        return Graph._from_pb(self._ir.getGraph(_id))

    @property
    def forward_graph(self):
        """Get the forward graph that was differentiated."""
        return Graph._from_pb(self._fwd_graph)

    @property
    def expected_inputs(self) -> Tuple[ExpectedConnection, ...]:
        """
        Get information about all expected inputs of the gradient graph.

        Inputs are tensors or their gradients from the forward graph
        that are required as inputs to the gradient graph.
        """
        return tuple(self._expected_inputs)

    @property
    def expected_outputs(self) -> Tuple[ExpectedConnection, ...]:
        """
        Get information about all expected outputs of the gradient graph.

        Outputs are tensors from the forward graph that have gradients as
        outputs of the gradient graph.
        """
        return tuple(self._expected_outputs)

    def __repr__(self) -> str:
        """
        Get a string representation for :py:class:`popxl.GradGraphInfo`.
        """
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
        """Get a tuple of all expected inputs for the gradient graph."""
        return tuple(map(lambda ec: ec.fwd_tensor, self._expected_inputs))

    @property
    def outputs(self) -> Tuple[Tensor, ...]:
        """Get a tuple of all expected outputs of the gradient graph."""
        return tuple(map(lambda ec: ec.fwd_tensor, self._expected_outputs))

    def inputs_dict(self, fwd_call_info: CallSiteInfo) -> Dict[Tensor, Tensor]:
        """
        Create mapping between gradient graph inputs to parent graph inputs.

        The mapping between the inputs to the gradient graph and the inputs to
        the parent graph is created from the forward graph's call site
        information.

        .. note:: This provides an easy way to pass activations to the gradient
           graph. It does not handle the gradient inputs of the gradient graph.

        Example:

        .. code-block:: python

            # `module`: subgraph module, `x`: parent graph inputs, `x_dash`: gradient graph parent input
            graph = ir.create_graph(module, x, out_features=16) # Forward graph
            call_info = ops.call_with_info(
                graph, x, inputs_dict={module.W: W, module.b: b})

            grads_graph = popxl.transforms.autodiff(graph)
            activations = ss_bwd_info.inputs_dict(call_info)
            grads_call_info = ops.call_with_info(
                grads_graph, x_dash, inputs_dict=activations)

        This method raises a ``TypeError`` exception if the forward graph does
        not match the graph that was differentiated.

        Args:
            fwd_call_info (CallSiteInfo):
                Call site information of a call to the forward graph that was
                differentiated. This can be accessed with
                :py:class:`popxl.ops.call_with_info()`.

        Raises:
            TypeError: If the called graph does not match the graph that was auto-differentiated.

        Returns:
            Dict[Tensor, Tensor]:
                A dictionary that maps from a tensor in the gradient graph to an
                input or output parent graph tensor at a call site of the
                corresponding forward graph.
        """
        if fwd_call_info.called_graph != self.forward_graph:
            raise TypeError(
                "The called graph does not match the graph that was differentiated."
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
        Return mapping between forward graph inputs and outputs of the parent of the gradient graph.

        ``autodiff`` is applied to the forward graph. This method returns the
        mapping between the input tensors of the forward graph and the output
        tensors of the parent of the gradient graph.

        Example:

        .. code-block:: python

            # `module`: subgraph module, `x`: parent graph inputs, `x_dash`: gradient parent graph input
            graph = ir.create_graph(module, x, out_features=16) # Forward graph
            call_info = ops.call_with_info(
                graph, x, inputs_dict={module.W: W, module.b: b})

            grads_graph = popxl.transforms.autodiff(graph)
            activations = ss_bwd_info.inputs_dict(call_info)
            grads_call_info = ops.call_with_info(
                grads_graph, x_dash, inputs_dict=activations)

            # Obtain a mapping between subgraph tensors that correspond to `x`, `W` and `b` and the corresponding parent gradient tensors outputs
            grad_tensor_map = grads_graph.fwd_graph_ins_to_grad_parent_outs(grads_call_info)
            assert all(t in graph for t in grad_tensor_map.keys())
            assert all(t in main for t in grad_tensor_map.values())
            assert [t.id for t in grad_tensor_map.keys()] == [
                'Module_subgraph(0)/x', 'Module_subgraph(0)/W',
                'Module_subgraph(0)/b']
            assert [t.id for t in grad_tensor_map.values()] == [
                'Gradient___x', 'Gradient___W', 'Gradient___b']

        Args:
            grad_call_info (CallSiteInfo):
                Call site information of the forward graph. This is the graph
                that ``autodiff`` is applied to. This can be accessed with
                :py:class:`popxl.ops.call_with_info()`

        Returns:
            Dict[Tensor, Tensor]:
                A dictionary that maps from an input tensor in the forward
                graph to an output tensor in the parent of the gradient graph.
        """
        return {
            t: grad_call_info.graph_to_parent(self.graph.outputs[idx])
            for idx, t in enumerate(self.outputs)
        }

    def fwd_parent_ins_to_grad_parent_outs(
            self, fwd_call_info: CallSiteInfo,
            grad_call_info: CallSiteInfo) -> Dict[Tensor, Tensor]:
        """
        Return mapping between forward's parent graph inputs and gradient's parent graph outputs.

        ``autodiff`` is applied to the forward graph. This method returns the
        mapping between the input tensors from the parent of the forward graph
        and the output tensors of the parent of the gradient graph.

        Example:

        .. code-block:: python

            # `module`: subgraph module, `x`: graph inputs, `x_dash`: gradient graph input
            graph = ir.create_graph(module, x, out_features=16) # Forward graph
            call_info = ops.call_with_info(
                graph, x, inputs_dict={module.W: W, module.b: b})

            grads_graph = popxl.transforms.autodiff(graph)
            activations = ss_bwd_info.inputs_dict(call_info)
            grads_call_info = ops.call_with_info(
                grads_graph, x_dash, inputs_dict=activations)

            # Obtain a mapping between input tensors `x`, `W` and `b` and the corresponding gradient tensors
            grad_tensor_map = grads_graph.fwd_parent_ins_to_grad_parent_outs(call_info, grads_call_info)
            assert [t.id for t in grad_tensor_map.keys()] == ['x', 'W', 'b']

        Args:
            fwd_call_info (CallSiteInfo):
                Call site information of the forward graph that was
                differentiated. This can be accessed with
                :py:class:`popxl.ops.call_with_info()`
            grad_call_info (CallSiteInfo):
                Call site information of the associated gradient
                graph. This can be accessed with
                :py:class:`popxl.ops.call_with_info()`

        Returns:
            Dict[Tensor, Tensor]:
                A dictionary that maps from an input tensor in the parent of the
                forward graph to an output tensor in the parent of the gradient
                graph.
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
    Perform automatic differentiation of a graph.

    ``graph`` will be differentiated using the chain rule starting from
    ``grads_provided``. The outputs of the returned graph will be the gradients
    of the tensors in ``grads_required``. By default, ``grads_provided`` will be
    all of the outputs of the forward graph and ``grads_required`` will be all
    of the inputs to the forward graph.

    Any tensors in the forward graph that are needed to compute the gradients
    will be added as outputs to the forward graph (if not already an input or
    output).

    The returned :py:class:`popxl.GradGraphInfo` object contains information
    about the inputs (``expected_inputs``) and outputs (``expected_outputs``) of
    the gradient graph. These are lists of tuples where the first element is
    either :py:class:`popxl.ExpectedConnectionType.Fwd` or
    :py:class:`popxl.ExpectedConnectionType.FwdGrad` meaning the input or output
    is associated with a tensor in the forward graph, or the gradient of a
    tensor in the forward graph, respectively. The second element is a tensor of
    the forward graph itself. These tensors are guaranteed to be either inputs
    or outputs of the forward graph.

    The ``expected_inputs`` list that describes the gradient graph's inputs is
    guaranteed to start with :py:class:`popxl.ExpectedConnectionType.FwdGrad`
    entries that exactly match the order of the entries in ``grads_provided``.

    ``expected_outputs`` describes the gradient graph's outputs and is
    guaranteed to comprise only :py:class:`popxl.ExpectedConnectionType.FwdGrad`
    entries and has entries that exactly match the size and order of
    ``grads_required``.

    Any subgraphs called from the forward graph will recursively have
    ``autodiff`` called on them. `return_all_grad_graphs` can be set to ``True``
    to return information on all graphs that the ``autodiff`` transform has been
    recursively applied to. Then, `called_graphs_grad_info` can be used to pass
    these previously calculated gradients to ``autodiff`` for those subgraphs.
    By default, :py:class:`popxl.GradGraphInfo` will only be returned for the
    input forward graph.

    Args:
        graph (Graph):
            The graph to differentiate.
            grads_provided (Optional[Iterable[popxl.Tensor]], optional):
            The list of gradients that are available for ``autodiff`` to use.
            Defaults to all outputs of the graph.
        grads_required (Optional[Iterable[popxl.Tensor]], optional):
            The list of inputs of the forward graph for which gradients are
            required. Defaults to all inputs of the graph.
        grads_provided (Optional[Iterable[popxl.Tensor]], optional):
            The list of inputs of the forward graph for which gradients are
            provided. Defaults to all inputs of the graph.
        called_graphs_grad_info (Optional[Mapping[Graph,GradGraphInfo]], optional):
            The gradient graph information for the subgraphs that the
            ``autodiff`` transform has been recursively applied to. Defaults to
            None.
        return_all_grad_graphs (bool, optional):
            Indicates whether to return the gradient graph information for all
            the subgraphs that the ``autodiff`` transform has been recursively
            applied to (``True``) or to only return the gradient graph
            information for ``graph`` (``False``). Defaults to ``False``.

    Returns:
        :py:class:`popxl.GradGraphInfo`:
            Information about the gradient graph.
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
