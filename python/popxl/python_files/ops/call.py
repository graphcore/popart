# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Mapping, Union, Tuple, Optional, Iterable, List, Dict

import popart._internal.ir as _ir
from popxl.context import (
    get_current_context,
    debug_context_frame_offset,
    op_debug_context,
)
from popxl.graph import Graph
from popxl.tensor import Tensor, TensorLike

from popxl.ops.utils import check_in_graph
from popxl.utils import table_to_string


# TODO: Bind subgraph op T53714
class CallSiteInfo:
    """
    Information relating to a parent graph calling a subgraph, for example using a call op or repeat op.

    This is a convenience class for extracting information about the callsite and its subgraph.
    """

    def __init__(self, subgraph_op: Union[_ir.op.CallOp, _ir.op.LoopOp]):
        self._op = subgraph_op

    @property
    def called_graph(self):
        return Graph._from_pb(self._op.getCalledGraphs()[0])

    def graph_to_parent_input_index(self, idx: int) -> int:
        """Get the parent graph input tensor index given the graph input tensor index."""
        return self._op.subgraphInToOpInIndex(idx)

    def parent_to_graph_input_index(self, idx: int) -> int:
        """Get the graph input tensor index given the parent graph input tensor index."""
        return self._op.opInToSubgraphInIndex(idx)

    def graph_to_parent_output_index(self, idx: int) -> int:
        """Get the parent graph output tensor index given the graph output tensor index."""
        return self._op.subgraphOutToOpOutIndex(idx)

    def parent_to_graph_output_index(self, idx: int) -> int:
        """Get the graph output tensor index given the parent graph output tensor index."""
        return self._op.opOutToSubgraphOutIndex(idx)

    def graph_to_parent(self, graph_tensor: Tensor) -> Tensor:
        """
        Get the tensor in the parent graph using the tensor in the called graph.

        Both input and output tensors can be used

        Args:
            graph_tensor (Tensor): The tensor in the called graph.

        Raises:
            ValueError: If `graph_tensor` is not an input or output of the called graph.

        Returns:
            Tensor: The associated input or output tensor on the `CallOp`.
        """
        sgraph = self.called_graph._pb_graph
        if sgraph.hasInputId(graph_tensor.id):
            idx = sgraph.getInputIndex(graph_tensor.id)
            return Tensor._from_pb_tensor(self._op.inTensor(idx))
        if sgraph.hasOutputId(graph_tensor.id):
            idx = sgraph.getOutputIndex(graph_tensor.id)
            return Tensor._from_pb_tensor(self._op.outTensor(idx))
        raise ValueError(
            f"Tensor {graph_tensor.name} is not an input or output of the called graph {sgraph.id}"
        )

    def parent_to_graph(self, parent_tensor: Tensor) -> Tensor:
        """
        Get the input tensor in the called graph using the input tensor in the parent graph.

        Args:
            parent_tensor (Tensor): The tensor from the parent graph.

        Raises:
            popart_error: If `parent_tensor` is not an input to the CallOp.

        Returns:
            Tensor: The tensor in the `called_graph`.
        """
        pb_subgraph = self.called_graph._pb_graph

        # Throws if not an input
        op_in_idx = self._op.firstInIndex(parent_tensor._pb_tensor)
        pb_sub_tensor = pb_subgraph.getInputTensor(
            self._op.opInToSubgraphInIndex(op_in_idx)
        )

        return Tensor._from_pb_tensor(pb_sub_tensor)

    def parent_input(self, idx: int) -> Tensor:
        """Get the parent graph input tensor at a given index."""
        pb_op_in_tensor = self._op.inTensor(idx)
        return Tensor._from_pb_tensor(pb_op_in_tensor)

    def parent_output(self, idx: int) -> Tensor:
        """Get the parent graph output tensor at a given index."""
        pb_op_out_tensor = self._op.outTensor(idx)
        return Tensor._from_pb_tensor(pb_op_out_tensor)

    @property
    def inputs(self) -> Tuple[Tensor, ...]:
        """
        Get the parent graph inputs.

        Returns:
            Tuple[Tensor, ...]
        """
        return tuple(Tensor._from_pb_tensor(t) for t in self._op.getInputTensors())

    @property
    def outputs(self) -> Tuple[Tensor, ...]:
        """
        Get the parent graph outputs.

        Returns:
            Tuple[Tensor, ...]
        """
        return tuple(Tensor._from_pb_tensor(t) for t in self._op.getOutputTensors())

    def set_parent_input_modified(
        self, parent_tensor: Tensor, infer_modified_regions: bool = True
    ):
        """
        Specify that the parent graph's input tensor `parent_tensor` is modified by the call op.

        This will guarantee that any modification to the graph input during the execution
        of the called graph will also change `parent_tensor`.

        Args:
            parent_tensor (Tensor): Input tensor in parent graph to be modified.
            infer_modified_regions (bool): Whether or not to infer the modified regions.

        Args:
            op_tensor (Tensor): Tensor to be modified.
            infer_modified_regions (bool): Set the modified regions from the Ops in the called graph.
        """
        index = self._op.inIndex(parent_tensor._pb_tensor)
        _graph = self._op.getCalledGraph()
        _sg_tensor = _graph.getInputTensor(self.parent_to_graph_input_index(index))
        if infer_modified_regions:
            _regions = _sg_tensor.modifiedRegionsByOps(_graph.getOps())
        else:
            _regions = [_ir.view.Region.getFull(parent_tensor.shape)]
        self._op.addModified(index, _regions)


@debug_context_frame_offset(1)
def call(
    graph: Graph,
    *inputs: Union[TensorLike, Iterable[TensorLike]],
    inputs_dict: Optional[Mapping[Tensor, TensorLike]] = None,
) -> Tuple[Tensor, ...]:
    """
    Call a graph.

    The `inputs` and `inputs_dict` tensors are passed as graph inputs.
    You can specify an input either positionally using `inputs` or via a tensor
    map using `inputs_dict`.

    Graph inputs are determined when the graph was created using `ir.create_graph(callable, ...)`.
    The order of inputs in will be the same as the order of the tensor inputs in the
    function signature and the order of called `popxl.graph_inputs`.

    See  :py:meth:`~popxl.Ir.create_graph` for more information.

    Args:
        graph (Graph): The graph to call.
        inputs (Union[TensorLike, Iterable[TensorLike]]):
            Provide inputs via position.
        inputs_dict (Optional[Mapping[Tensor, TensorLike]]):
            Provide inputs via graph tensor. Mapping of `graph tensor -> parent tensor`.
    Returns:
        Tuple[Tensor, ...]:
            Tuple of the output tensors of the call in the parent graph.
    """
    info = call_with_info(graph, *inputs, inputs_dict=inputs_dict)
    return info.outputs


@op_debug_context("call")
def call_with_info(
    graph: Graph,
    *inputs: Union[TensorLike, Iterable[TensorLike]],
    inputs_dict: Optional[Mapping[Tensor, TensorLike]] = None,
    check_inputs: bool = True,
) -> CallSiteInfo:
    """
    Call a graph and return information about the call site.

    The `inputs` and `inputs_dict` tensors are passed as graph inputs.
    You can specify an input either positionally using `inputs` or via a tensor
    map using `inputs_dict`. This op returns `CallSiteInfo` that can be used
    to inspect call site inputs/outputs.

    Graph inputs are determined when the graph was created using `ir.create_graph(callable, ...)`.

    The order of inputs will be the same as the order of the tensor inputs in the
    function signature and the order of called `popxl.graph_inputs`.

    See  :py:meth:`~popxl.Ir.create_graph` for more information.

    Args:
        graph (Graph): The graph to call.
        inputs (Union[TensorLike, Iterable[TensorLike]]):
            Provide inputs via position.
        inputs_dict (Optional[Mapping[Tensor, TensorLike]]):
            Provide inputs via graph tensor. Mapping of `graph tensor -> parent tensor`.
        check_inputs (bool):
            Check when called if all inputs have been provided. Defaults to True.

    Raises:
        ValueError: A ValueError will be raised if:
            - An incorrect number of inputs have been provided
            - A parent input tensor is not in the parent graph
            - A graph input tensor is specified twice
        TypeError: A TypeError will be raised if:
            - Graph input tensor is specified twice
            - If a graph input cannot be coerced into a tensor

    Returns:
        CallSiteInfo:
            Information about the call site.
    """

    ctx = get_current_context()
    parent_graph = ctx.graph
    pb_g = parent_graph._pb_graph
    pb_sg = graph._pb_graph

    op_name = parent_graph.name + "--call--" + graph.name

    opid = _ir.OperatorIdentifier("ai.graphcore", "Call", 1, _ir.NumInputs(), 0)

    pb_callop = pb_g.createOp_CallOp(
        opid, graph._pb_graph, ctx._get_op_settings(op_name)
    )

    inputs, inputs_dict = _prep_inputs_pre_validation(inputs, inputs_dict)
    _validate_inputs(parent_graph, graph, inputs, inputs_dict)
    inputs, inputs_dict = _prep_inputs(graph, inputs, inputs_dict)

    # 1. Connect tensors passed positionally via `inputs`
    connected_inputs: List[Dict] = []
    for graph_input_idx, parent_tensor in enumerate(inputs):
        call_in_idx = pb_callop.subgraphInToOpInIndex(graph_input_idx)
        pb_callop.connectInTensor(call_in_idx, parent_tensor.id)
        connected_inputs += [
            {"graph_input_idx": graph_input_idx, "parent_tensor": parent_tensor}
        ]

    # 2. Connect tensors passed via a dict via `inputs_dict`
    for graph_tensor, parent_tensor in inputs_dict.items():
        graph_input_idx = pb_sg.getInputIndex(graph_tensor.id)
        call_in_idx = pb_callop.subgraphInToOpInIndex(graph_input_idx)
        pb_callop.connectInTensor(call_in_idx, parent_tensor.id)
        connected_inputs += [
            {"graph_input_idx": graph_input_idx, "parent_tensor": parent_tensor}
        ]

    _validate_connected_tensors(
        check_inputs, connected_inputs, graph, inputs, inputs_dict
    )

    # 3. Connect outputs. We introspect the subgraph to get its outputs then,
    #    for each one, create an output tensor of the call op in the parent
    #    graph.

    def id_like_subgraph_tensor(tensor_id: str) -> str:
        return parent_graph._create_tensor_id(_ir.removeScope(pb_sg, tensor_id))

    for sg_out_idx, pb_sg_out_id in enumerate(pb_sg.getOutputIds()):
        call_out_idx = pb_callop.subgraphOutToOpOutIndex(sg_out_idx)
        parent_tensor_id = id_like_subgraph_tensor(pb_sg_out_id)
        pb_callop.createAndConnectOutTensor(call_out_idx, parent_tensor_id)

    pb_callop.setup()

    info = CallSiteInfo(pb_callop)

    # 4. If marked as TensorByRef then set_parent_input_modified
    for parent_tensor in graph._by_ref_inputs:
        info.set_parent_input_modified(info.graph_to_parent(parent_tensor))

    return info


# Internal function shared with ops.repeat
def _prep_inputs_pre_validation(
    inputs: Iterable[Union[TensorLike, Iterable[TensorLike]]],
    inputs_dict: Dict[Tensor, TensorLike],
):
    inputs_dict = inputs_dict if inputs_dict is not None else {}

    # Flatten inputs
    inputs_flat: List[TensorLike] = []
    for x in inputs:
        if isinstance(x, Iterable) and not isinstance(x, str):
            inputs_flat.extend(x)
        else:
            inputs_flat.append(x)

    return inputs_flat, inputs_dict


# Internal function shared with ops.repeat
def _validate_inputs(
    parent_graph: Graph,
    graph: Graph,
    inputs: List[TensorLike],
    inputs_dict: Dict[Tensor, TensorLike],
):
    if len(inputs) > len(graph.inputs):
        raise ValueError(
            f"Too many inputs have been provided. The number of positional inputs exceeds the number of inputs."
            f"{len(inputs)} > {len(graph.inputs)}"
        )

    for graph_input_idx, parent_tensor in enumerate(inputs):
        if isinstance(parent_tensor, Tensor):
            try:
                check_in_graph(parent_graph, parent_tensor=parent_tensor)
            except ValueError:
                raise ValueError(
                    f"Parent input tensor {parent_tensor} is not in parent graph {parent_graph}."
                )

    for graph_tensor, parent_tensor in inputs_dict.items():
        try:
            check_in_graph(graph, graph_tensor=graph_tensor)
        except ValueError:
            raise ValueError(
                f'The graph input tensor "{graph_tensor}" is not in the called graph "{graph}".'
            )

        if isinstance(parent_tensor, Tensor):
            try:
                check_in_graph(parent_graph, parent_tensor=parent_tensor)
            except ValueError:
                raise ValueError(
                    f"The parent input tensor {parent_tensor} is not in the parent graph {parent_graph}."
                )

        graph_input_idx = graph._pb_graph.getInputIndex(graph_tensor.id)
        if graph_input_idx in range(len(inputs)):
            raise ValueError(
                f"Graph input tensor is specified twice, in `inputs_dict` and positionally in `inputs`: {graph_tensor}"
            )


# Internal function shared with ops.repeat
def _prep_inputs(
    graph: Graph, inputs: List[TensorLike], inputs_dict: Dict[Tensor, TensorLike]
):
    for graph_input_idx, parent_tensor in enumerate(inputs):
        if not isinstance(parent_tensor, Tensor):
            graph_tensor = graph.inputs[graph_input_idx]
            try:
                parent_tensor = graph_tensor._ensure_tensor(
                    parent_tensor
                )  # Coerce input into tensor of same dtype
            except ValueError:
                raise TypeError(
                    f"Input at position {graph_input_idx} could not be coerced into a tensor: Type {type(parent_tensor)}. Value: {parent_tensor}"
                )
            inputs[graph_input_idx] = parent_tensor

    for graph_tensor, parent_tensor in inputs_dict.items():
        if not isinstance(parent_tensor, Tensor):
            try:
                parent_tensor = graph_tensor._ensure_tensor(
                    parent_tensor
                )  # Coerce input into tensor of same dtype
            except ValueError:
                raise TypeError(
                    f"Input ({graph_tensor}) could not be coerced into a tensor: Type {type(parent_tensor)}. Value: {parent_tensor}"
                )
            inputs_dict[graph_tensor] = parent_tensor

    return inputs, inputs_dict


# Internal function shared with ops.repeat
def _validate_connected_tensors(
    check_inputs: bool,
    connected_inputs: List[Dict],
    graph: Graph,
    inputs: List[Tensor],
    inputs_dict: Dict[Tensor, Tensor],
):
    # Check if any tensors are missing.
    # Dont need to check if too many tensors are provided as previous "input specified twice" check should prevent that
    if check_inputs and len(inputs) + len(inputs_dict) < len(graph.inputs):
        connected_tensors = set(
            graph.inputs[d["graph_input_idx"]] for d in connected_inputs
        )
        connected_inputs = sorted(connected_inputs, key=lambda d: d["graph_input_idx"])
        missing_inputs_tb = [("Index", "Graph Tensor")] + [
            (idx, t.id)
            for idx, t in enumerate(graph.inputs)
            if t not in connected_tensors
        ]
        connected_inputs_tb = [("Index", "Graph Tensor", "Parent Graph Tensor")] + [
            (
                d["graph_input_idx"],
                graph.inputs[d["graph_input_idx"]].id,
                d["parent_tensor"].id,
            )
            for d in connected_inputs
        ]

        msg = "Not enough inputs have been provided.\n\nMissing inputs:\n"
        msg += table_to_string(missing_inputs_tb)
        msg += "\n\nConnected inputs:\n"
        msg += table_to_string(connected_inputs_tb)
        raise ValueError(msg)
