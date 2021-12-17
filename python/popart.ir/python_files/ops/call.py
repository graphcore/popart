# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Mapping, Union, Tuple

import popart._internal.ir as _ir
from popart.ir.context import get_current_context, debug_context_frame_offset, op_debug_context
from popart.ir.graph import Graph
from popart.ir.tensor import Tensor

from .utils import check_in_graph

from typing import Mapping, Union, Tuple, Optional, List


class CallInfo:
    def __init__(self, call_op: _ir.op.CallOp):
        self._op = call_op

    @property
    def called_graph(self):
        return Graph._from_pb(self._op.getCalledGraphs()[0])

    def subgraph_in_to_op_in_index(self, idx: int) -> Tensor:
        return self._op.subgraphInToOpInIndex(idx)

    def op_in_to_subgraph_in_index(self, idx: int) -> Tensor:
        return self._op.opInToSubgraphInIndex(idx)

    def subgraph_out_to_op_out_index(self, idx: int) -> Tensor:
        return self._op.subgraphOutToOpOutIndex(idx)

    def op_out_to_subgraph_out_index(self, idx: int) -> Tensor:
        return self._op.opOutToSubgraphOutIndex(idx)

    def subgraph_to_op_tensor(self, subgraph_tensor: Tensor) -> Tensor:
        """
        Provided an input or output tensor in the `called_graph`, this method
        returns the associated input or output tensor on the CallOp.

        Args:
            subgraph_tensor (Tensor): The tensor in the subgraph.

        Raises:
            ValueError: If `subgraph_tensor` is not an input or output of the called graph.

        Returns:
            Tensor: The associated input or output tensor on the CallOp
        """
        sgraph = self.called_graph._pb_graph
        if sgraph.hasInputId(subgraph_tensor.id):
            idx = sgraph.getInputIndex(subgraph_tensor.id)
            return Tensor._from_pb_tensor(self._op.inTensor(idx))
        if sgraph.hasOutputId(subgraph_tensor.id):
            idx = sgraph.getOutputIndex(subgraph_tensor.id)
            return Tensor._from_pb_tensor(self._op.outTensor(idx))
        raise ValueError(
            f"Tensor {subgraph_tensor.name} is not an Input or Output of the called graph {sgraph.id}"
        )

    def op_in_to_subgraph_in_tensor(self, parent_tensor: Tensor) -> Tensor:
        """
        Provided an input tensor on the CallOp, this method returns the
        associated input tensor in the `called_graph`.

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
            self._op.opInToSubgraphInIndex(op_in_idx))

        return Tensor._from_pb_tensor(pb_sub_tensor)

    def get_op_input_tensor(self, op_in_idx: int) -> Tensor:
        pb_op_in_tensor = self._op.inTensor(op_in_idx)
        return Tensor._from_pb_tensor(pb_op_in_tensor)

    def get_op_output_tensor(self, op_out_idx: int) -> Tensor:
        pb_op_out_tensor = self._op.outTensor(op_out_idx)
        return Tensor._from_pb_tensor(pb_op_out_tensor)

    def get_input_tensors(self) -> Tuple[Tensor, ...]:
        """Return inputs to the CallOp in index order.

        Returns:
            Tuple[Tensor, ...]
        """
        return tuple(
            Tensor._from_pb_tensor(t) for t in self._op.getInputTensors())

    def get_output_tensors(self) -> Tuple[Tensor, ...]:
        """Return outputs to the CallOp in index order.

        Returns:
            Tuple[Tensor, ...]
        """
        return tuple(
            Tensor._from_pb_tensor(t) for t in self._op.getOutputTensors())

    def set_op_input_modified(self, op_tensor: Tensor):
        """Specify that the input tensor `op_tensor` is modified by the call op.
            this will guarentee that any modification to the graph input during the execution
            of the called graph will also change `op_tensor`.
            The regions modified by the call op will be specified by the Ops in the called graph.

        Args:
            op_tensor (Tensor): Tensor to be modified.
        """
        index = self._op.inIndex(op_tensor._pb_tensor)
        _graph = self._op.getCalledGraph()
        _sg_tensor = _graph.getInputTensor(
            self.op_in_to_subgraph_in_index(index))
        _regions = _sg_tensor.modifiedRegionsByOps(_graph.getOps())
        self._op.addModified(index, _regions)


@debug_context_frame_offset(1)
def call(subgraph: Graph,
         *subgraph_fn_param_inputs: Union[Tensor, List[Tensor]],
         subgraph_in_to_parent_in: Optional[Mapping[Tensor, Tensor]] = None
         ) -> Union[None, Tensor, Tuple[Tensor, ...]]:
    """
    Call Op: An op that invokes a subgraph with the provided input tensors.

    Args:
        subgraph (Graph): The called graph.
        *subgraph_fn_param_inputs  (Tensor, List[Tensor]):
            parent tensors that correspond to the inputs of the callable passed
            to ir.create_graph(callable, ...) when constructing `subgraph` earlier.
            The inputs passed MUST be provided here in the EXACT SAME ORDER as
            to ir.create_graph(callable, ...).
        subgraph_in_to_parent_in (Mapping[Tensor, Tensor] = {}):
            Mapping of `subgraph tensor -> parent tensor` that corresponds to
            the inputs that the callable defined internally, e.g. by using
            popart.ir.subgraph_input. Defaults to an empty dictionary.

    Returns:
        None: If `subgraph` has no output tensors.
        Tensor:
            The output tensor of the call in the parent graph, if #subgraph has
            exactly 1 output.
        Tuple[Tensor, ...]:
            Tuple of the output tensors of the call in the parent graph, if
            `subgraph` has >1 outputs. The tensors will be in ascending order of
            the graph output index of the corresponding subgraph tensor.
    """
    info = call_with_info(subgraph,
                          *subgraph_fn_param_inputs,
                          subgraph_in_to_parent_in=subgraph_in_to_parent_in)
    out_tensors = info.get_output_tensors()

    # Return nothing if no outputs.
    if len(out_tensors) == 0:
        return
    # Return single tensor if only one output.
    if len(out_tensors) == 1:
        return out_tensors[0]
    # Return tuple of output tensors if multiple outputs.
    return out_tensors


@op_debug_context("call")
def call_with_info(
        subgraph: Graph,
        *subgraph_fn_param_inputs: Union[Tensor, List[Tensor]],
        subgraph_in_to_parent_in: Optional[Mapping[Tensor, Tensor]] = None,
        check_inputs: bool = True,
) -> CallInfo:
    """
    Call Op: An op that invokes a subgraph with the provided input tensors.
        Returns CallInfo that can be used to inspect callsite inputs/outputs.

    Args:
        subgraph (Graph): The called graph.
        *subgraph_fn_param_inputs (Tensor, List[Tensor]):
            parent tensors that correspond to the inputs of the callable passed
            to ir.create_graph(callable, ...) when constructing `subgraph` earlier.
            The inputs passed MUST be provided here in the EXACT SAME ORDER as
            to ir.create_graph(callable, ...).
        subgraph_in_to_parent_in (Mapping[Tensor, Tensor] = {}):
            Mapping of `subgraph tensor -> parent tensor` that corresponds to
            the inputs that the callable defined internally, e.g. by using
            popart.ir.subgraph_input. Defaults to an empty dictionary.
        check_inputs (bool = True):
            Check when called if all inputs have been provided.
    Returns:
        info: CallInfo
            Information on the created callsite.
    """
    subgraph_in_to_parent_in = subgraph_in_to_parent_in if subgraph_in_to_parent_in is not None else {}

    subgraph_fn_param_inputs_flat = []
    for x in subgraph_fn_param_inputs:
        if isinstance(x, (list, tuple)):
            subgraph_fn_param_inputs_flat.extend(x)
        else:
            subgraph_fn_param_inputs_flat.append(x)

    if check_inputs and len(subgraph_fn_param_inputs_flat) + len(
            subgraph_in_to_parent_in) != len(subgraph.get_input_tensors()):
        raise ValueError(
            "Not enough inputs have been provided: the number of graph inputs does not equal the number of "
            "subgraph_fn_param_inputs plus subgraph_in_to_parent_in inputs: {} != {} + {}"
            .format(len(subgraph.get_input_tensors()),
                    len(subgraph_fn_param_inputs_flat),
                    len(subgraph_in_to_parent_in)))

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    pb_sg = subgraph._pb_graph

    op_name = g.name + '--call--' + subgraph.name

    opid = _ir.OperatorIdentifier("ai.graphcore", "Call", 1, _ir.NumInputs(),
                                  0)

    pb_callop = pb_g.createOp_CallOp(opid, subgraph._pb_graph,
                                     ctx._get_op_settings(op_name))

    # 1. Connect explicitly passed inputs. These would have been created first
    #    by ir.create_graph, so we do them first. ir.create_graph will have created
    #    the input tensors t_0,...,t_N at input indices 0,..,N, respectively. We
    #    require that the user has passed the parent tensors that correspond to
    #    these inputs in the exact same order, so we can trivially reconstruct
    #    the input indices here.
    sgInIdx = 0
    for t in subgraph_fn_param_inputs_flat:
        callInIdx = pb_callop.subgraphInToOpInIndex(sgInIdx)
        pb_callop.connectInTensor(callInIdx, t.id)
        sgInIdx += 1

    # 2. Connect internally created inputs.
    for sg_tensor, parent_tensor in subgraph_in_to_parent_in.items():
        check_in_graph(g, parent_tensor)
        check_in_graph(subgraph, sg_tensor)

        sgInIdx = pb_sg.getInputIndex(sg_tensor.id)
        callInIdx = pb_callop.subgraphInToOpInIndex(sgInIdx)
        pb_callop.connectInTensor(callInIdx, parent_tensor.id)

    # 3. Connect outputs. We introspect the subgraph to get its outputs then,
    #    for each one, create an output tensor of the call op in the parent
    #    graph.

    def id_like_subgraph_tensor(tensor_id: str) -> str:
        return g._create_tensor_id(_ir.removeScope(pb_sg, tensor_id))

    for pb_sg_out_id in pb_sg.getOutputIds():
        sgOutIdx = pb_sg.getOutputIndex(pb_sg_out_id)
        callOutIdx = pb_callop.subgraphOutToOpOutIndex(sgOutIdx)
        parent_tensor_id = id_like_subgraph_tensor(pb_sg_out_id)
        pb_callop.createAndConnectOutTensor(callOutIdx, parent_tensor_id)

    pb_callop.setup()

    info = CallInfo(pb_callop)

    for t in subgraph._by_ref_inputs:
        info.set_op_input_modified(info.subgraph_to_op_tensor(t))

    return info
