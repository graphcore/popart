# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Iterable, Mapping, Optional, Tuple, Union

from popart import popart_exception
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.graph import Graph
from popxl.tensor import Tensor
from popxl.ops.call import _prep_and_validate_inputs, CallSiteInfo


class ConditionalSiteInfo(CallSiteInfo):
    """
    Information relating to a parent graph calling a subgraph, for example using a call op or repeat op.

    This is a convenience class for extracting information about the callsite and its subgraph.
    """

    @property
    def called_graph(self) -> Tuple[Graph, Graph]:
        """Get the called graphs of conditional op"""
        return Graph._from_pb(self._op.getCalledGraphs()[0]), Graph._from_pb(
            self._op.getCalledGraphs()[1])

    def graph_to_parent_input_index(self, idx: int,
                                    then_branch: bool = True) -> int:
        """Get the parent graph input tensor index given the graph input tensor index."""
        try:
            return self._op.subgraphInToOpInIndex(int(not then_branch), idx)
        except popart_exception as e:
            raise IndexError("Invalid graph input index") from e

    def parent_to_graph_input_index(self, idx: int) -> int:
        """Get the graph input tensor index given the parent graph input tensor index."""
        try:
            # Try then_branch, then try else_branch
            graph_idx = self._op.opInToSubgraphInIndex(0, idx)
            if graph_idx == -1:
                graph_idx = self._op.opInToSubgraphInIndex(1, idx)
            if graph_idx == -1:
                raise popart_exception('Invalid parent graph input index')
            return graph_idx
        except popart_exception as e:
            raise IndexError('Invalid parent graph input index') from e

    def is_parent_index_in_then_branch(self, idx: int) -> bool:
        """Determine if a parent input tensor index belongs to the `then_branch` graph or
           `else_branch`. Raises IndexError if invalid or belongs to neither."""
        try:
            # Try then_branch, then try else_branch
            graph_idx = self._op.opInToSubgraphInIndex(0, idx)
            if graph_idx != -1:
                return True
            graph_idx = self._op.opInToSubgraphInIndex(1, idx)
            if graph_idx != -1:
                return False
            raise popart_exception('Invalid parent graph input index')
        except popart_exception as e:
            raise IndexError('Invalid parent graph input index') from e

    def graph_to_parent_output_index(self, idx: int,
                                     then_branch: bool = True) -> int:
        """Get the parent graph output tensor index given the graph output tensor index.

        For a conditional op call site the "then" graph is selected when `then_branch` is True,
        otherwise the "else" graph is selected."""
        try:
            return self._op.subgraphOutToOpOutIndex(int(not then_branch), idx)
        except popart_exception as e:
            raise IndexError('Invalid graph input index') from e

    def parent_to_graph_output_index(self, idx: int,
                                     then_branch: bool = True) -> int:
        """Get the graph output tensor index given the parent graph output tensor index.

        For a conditional op call site the "then" graph is selected when `then_branch` is True,
        otherwise the "else" graph is selected."""
        try:
            return self._op.opOutToSubgraphOutIndex(int(not then_branch), idx)
        except popart_exception as e:
            raise IndexError('Invalid graph input index') from e

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
        called_graphs = self._op.getCalledGraphs()
        for graph, then_branch in zip(called_graphs, [True, False]):
            if graph.hasInputId(graph_tensor.id):
                idx = graph.getInputIndex(graph_tensor.id)
                parent_idx = self.graph_to_parent_input_index(idx, then_branch)
                return Tensor._from_pb_tensor(self._op.inTensor(parent_idx))
            if graph.hasOutputId(graph_tensor.id):
                idx = graph.getOutputIndex(graph_tensor.id)
                parent_idx = self.graph_to_parent_output_index(
                    idx, then_branch)
                return Tensor._from_pb_tensor(self._op.outTensor(parent_idx))
        raise ValueError(
            f"Tensor {graph_tensor.name} is not an input or output of the called graph(s)."
        )

    def parent_to_graph(self, parent_tensor: Tensor) -> Tensor:
        """
        Get the input tensor in the called graph using the input tensor in the parent graph.

        If the `parent_tensor` has been used multiple times as an input only the first instance
        is returned.

        Args:
            parent_tensor (Tensor): The tensor from the parent graph.

        Raises:
            popart_error: If `parent_tensor` is not an input to the CallOp.

        Returns:
            Tensor: The tensor in the `called_graph`.
        """
        try:
            op_in_idx = self._op.firstInIndex(parent_tensor._pb_tensor)
        except IndexError as e:
            raise ValueError(
                f"Tensor {parent_tensor.name} is not a parent graph input of the call site."
            ) from e
        graph_idx = self.parent_to_graph_input_index(op_in_idx)
        is_then_graph = self.is_parent_index_in_then_branch(op_in_idx)
        called_graphs = self._op.getCalledGraphs()
        graph = called_graphs[int(not is_then_graph)]
        graph_input_tensor = graph.getInputTensor(graph_idx)
        return Tensor._from_pb_tensor(graph_input_tensor)

    def set_parent_input_modified(self,
                                  parent_tensor: Tensor,
                                  infer_modified_regions: bool = True):
        raise NotImplementedError(
            'Parent input modification is not yet implemented for the conditional op.'
        )


@op_debug_context
def conditional(cond: Tensor,
                then_branch: Graph,
                else_branch: Graph,
                then_inputs: Optional[Iterable[
                    Union[Tensor, Iterable[Tensor]]]] = None,
                else_inputs: Optional[Iterable[
                    Union[Tensor, Iterable[Tensor]]]] = None,
                then_inputs_dict: Optional[Mapping[Tensor, Tensor]] = None,
                else_inputs_dict: Optional[Mapping[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, ...]:
    """
    Execute `then_branch` or `else_branch` according to the value of tensor `cond` at runtime.

    The `then/else_inputs` and `then/else_inputs_dict` tensors are passed as then/else_branch inputs.
    You can specify a then/else_input either positionally using `then/else_inputs` or via a tensor
    map using `then/else_inputs_dict`.

    Graph inputs are determined when the graph was created using `ir.create_graph(callable, ...)`.

    The order of inputs will be the same as the order of the tensor inputs in the
    function signature and the order of called `popxl.graph_inputs`.

    See  :py:meth:`~popxl.Ir.create_graph` for more information.

    Args:
        cond (Tensor):
            A boolean single-value tensor. If true the then_branch is executed otherwise the
            else_branch is executed.
        then_branch (Graph):
            Graph to run if condition is true.
        else_branch (Graph):
            Graph to run if condition is false.
        then_inputs (Optional[Iterable[Union[Tensor, Iterable[Tensor]]]]):
            Provide inputs to then_branch via position, `then_inputs` follow the same rules as `inputs` in
            `call` and `repeat` op.
        else_inputs (Optional[Iterable[Union[Tensor, Iterable[Tensor]]]]):
            Provide inputs to else_branch via position, `else_inputs` follow the same rules as `inputs` in 
            `call` and `repeat` op.
        then_inputs_dict (Optional[Mapping[Tensor, Tensor]]):
            Provide inputs to then_branch via a tensor map. Mapping of `graph tensor -> parent tensor`,
            `then_inputs_dict` follow the same rules as `inputs_dict` in `call` and `repeat` op.
        else_inputs_dict (Optional[Mapping[Tensor, Tensor]]):
            `else_inputs_dict` follow the same rules as `inputs_dict` in `call` and `repeat` op.
    
    Raises:
        ValueError: If:
            - An incorrect number of inputs have been provided.
            - A parent input tensor is not in the parent graph.
            - A graph input tensor is specified twice.
        TypeError: If:
            - A graph input tensor is specified twice.
            - A graph input cannot be coerced into a tensor.
    
    Returns:
        Tuple[Tensor, ...]:
            The values that are live after the execution of the conditional. The return values in `then_branch`
            and `else_branch` must be of the same data type. The number of the outputs in `then_branch` and 
            `else_branch` must be equal. The shape of the input and outputs in `then_branch` and `else_branch`
            must also be the same.
    """
    call_info = conditional_with_info(cond, then_branch, else_branch,
                                      then_inputs, else_inputs,
                                      then_inputs_dict, else_inputs_dict)

    return call_info.outputs


@op_debug_context
def conditional_with_info(
        cond: Tensor,
        then_branch: Graph,
        else_branch: Graph,
        then_inputs: Optional[Iterable[
            Union[Tensor, Iterable[Tensor]]]] = None,
        else_inputs: Optional[Iterable[
            Union[Tensor, Iterable[Tensor]]]] = None,
        then_inputs_dict: Optional[Mapping[Tensor, Tensor]] = None,
        else_inputs_dict: Optional[Mapping[Tensor, Tensor]] = None,
        check_inputs: bool = True) -> ConditionalSiteInfo:
    """
    Execute `then_branch` or `else_branch` according to the value of tensor `cond` at runtime
    and return the call site info.

    The `then/else_inputs` and `then/else_inputs_dict` tensors are passed as then/else_branch inputs.
    You can specify a then/else_input either positionally using `then/else_inputs` or via a tensor
    map using `then/else_inputs_dict`.

    Graph inputs are determined when the graph was created using `ir.create_graph(callable, ...)`.

    The order of inputs will be the same as the order of the tensor inputs in the
    function signature and the order of called `popxl.graph_inputs`.

    See  :py:meth:`~popxl.Ir.create_graph` for more information.

    Args:
        cond (Tensor):
            A boolean single-value tensor. If true the then_branch is executed otherwise the
            else_branch is executed.
        then_branch (Graph):
            Graph to run if condition is true.
        else_branch (Graph):
            Graph to run if condition is false.
        then_inputs (Optional[Iterable[Union[Tensor, Iterable[Tensor]]]]):
            Provide inputs to then_branch via position, `then_inputs` follow the same rules as `inputs` in
            `call` and `repeat` op.
        else_inputs (Optional[Iterable[Union[Tensor, Iterable[Tensor]]]]):
            Provide inputs to else_branch via position, `else_inputs` follow the same rules as `inputs` in 
            `call` and `repeat` op.
        then_inputs_dict (Optional[Mapping[Tensor, Tensor]]):
            Provide inputs to then_branch via a tensor map. Mapping of `graph tensor -> parent tensor`,
            `then_inputs_dict` follow the same rules as `inputs_dict` in `call` and `repeat` op.
        else_inputs_dict (Optional[Mapping[Tensor, Tensor]]):
            `else_inputs_dict` follow the same rules as `inputs_dict` in `call` and `repeat` op.
        check_inputs (bool):
            Check when called if all inputs have been provided to both graphs. Defaults to True.
    
    Raises:
        ValueError: If:
            - An incorrect number of inputs have been provided.
            - A parent input tensor is not in the parent graph.
            - A graph input tensor is specified twice.
        TypeError: If:
            - A graph input tensor is specified twice.
            - A graph input cannot be coerced into a tensor.
    
    Returns:
        ConditionalSiteInfo:
            Information on the created conditional site.
    """
    ctx = get_current_context()
    parent_graph = ctx.graph
    pb_graph = parent_graph._pb_graph
    pb_then_graph = then_branch._pb_graph
    pb_else_graph = else_branch._pb_graph

    # 1. Prep and validate inputs
    then_all_inputs = _prep_and_validate_inputs(check_inputs, parent_graph,
                                                then_branch, 'then_branch',
                                                then_inputs, then_inputs_dict)
    else_all_inputs = _prep_and_validate_inputs(check_inputs, parent_graph,
                                                else_branch, 'else_branch',
                                                else_inputs, else_inputs_dict)
    if_inputs = then_all_inputs + else_all_inputs

    # 2. Validate number of outputs
    if len(pb_then_graph.getOutputIds()) != len(pb_else_graph.getOutputIds()):
        raise ValueError(
            "`then_branch` graph and `else_branch` graph must have the same number of outputs: "
            f"{len(pb_then_graph.getOutputIds())} != {len(pb_else_graph.getOutputIds())}"
        )
    num_outputs = len(pb_then_graph.getOutputIds())

    # 3. Create op
    opid = _ir.OperatorIdentifier("ai.onnx", "If", 11, _ir.NumInputs(3, 3), 0)

    then_graph_id = _ir.GraphId(pb_then_graph.id.str())
    else_graph_id = _ir.GraphId(pb_else_graph.id.str())

    # Create inputs maps, +1 is due to the cond being the input at index 0
    then_input_indices_map = {i + 1: i for i in range(len(then_all_inputs))}
    else_input_indices_map = {
        i + 1 + len(then_all_inputs): i
        for i in range(len(else_all_inputs))
    }

    # Create output maps
    then_and_else_output_indices_map = {i: i for i in range(num_outputs)}

    # Set up IfOp
    then_branch_info = _ir.op.BranchInfo(then_graph_id, then_input_indices_map,
                                         then_and_else_output_indices_map)
    else_branch_info = _ir.op.BranchInfo(else_graph_id, else_input_indices_map,
                                         then_and_else_output_indices_map)
    pb_if_op = pb_graph.createOp_IfOp(opid, then_branch_info, else_branch_info,
                                      ctx._get_op_settings('conditional'))

    # 6. Connect input tensors
    pb_if_op.connectInTensor(pb_if_op.getConditionInIndex(), cond.id)
    for input_idx, parent_tensor in enumerate(if_inputs):
        pb_if_op.connectInTensor(input_idx + 1, parent_tensor.id)

    # 7. Connect output tensors
    for idx, (then_out_id, else_out_id) in enumerate(
            zip(pb_then_graph.getOutputIds(), pb_else_graph.getOutputIds())):
        then_parent_tensor_name = _ir.removeScope(pb_then_graph, then_out_id)
        else_parent_tensor_name = _ir.removeScope(pb_else_graph, else_out_id)
        output_name = f"{then_parent_tensor_name}_or_{else_parent_tensor_name}"
        output_id = parent_graph._create_tensor_id(output_name)
        pb_if_op.createAndConnectOutTensor(idx, output_id)

    # 8. Setup IfOp and return the outputs
    pb_if_op.setup()
    info = ConditionalSiteInfo(pb_if_op)

    # 9. If marked as TensorByRef then set_parent_input_modified
    # TODO T66328: implement TensorByRef for conditional op
    for graph_tensor in then_branch._by_ref_inputs:
        raise NotImplementedError(
            f"TensorByRef is not implemented for the conditional op. "
            f"You cannot call this input to the then_branch graph by ref: {graph_tensor}"
        )
        # TODO T66328
        # _regions = graph_tensor._pb_tensor.modifiedRegionsByOps(then_branch._pb_graph.getOps())
        # index = list(then_branch.inputs).index(graph_tensor)
        # pb_if_op.addModified(index, _regions)
    for graph_tensor in else_branch._by_ref_inputs:
        raise NotImplementedError(
            f"TensorByRef is not implemented for the conditional op. "
            f"You cannot call this input to the else_branch graph by ref: {graph_tensor}"
        )
        # TODO T66328
        # _regions = graph_tensor._pb_tensor.modifiedRegionsByOps(else_branch._pb_graph.getOps())
        # index = list(else_branch.inputs).index(graph_tensor)
        # pb_if_op.addModified(index, _regions)

    return info