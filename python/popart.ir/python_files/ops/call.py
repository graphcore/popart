# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.graph import Graph
from popart.ir.tensor import Tensor

from .utils import check_in_graph

from typing import Mapping, Union, Tuple, Optional


def call(subgraph: Graph,
         *subgraph_fn_param_inputs: Tensor,
         subgraph_in_to_parent_in: Optional[Mapping[Tensor, Tensor]] = None
         ) -> Union[None, Tensor, Tuple[Tensor]]:
    """
    Call Op: An op that invokes a subgraph with the provided input tensors.

    Args:
        subgraph (Graph): The called graph.
        *subgraph_fn_param_inputs (Tensor):
            parent tensors that correspond to the inputs of the callable passed
            to ir.get_graph(callable, ...) when constructing #subgraph earlier.
            The inputs passed MUST be provided here in the EXACT SAME ORDER as
            to ir.get_grah(callable, ...).
        subgraph_in_to_parent_in (Mapping[Tensor, Tensor] = {}):
            Mapping of `subgraph tensor -> parent tensor` that corresponds to
            the inputs that the callable defined internally, e.g. by using
            popart.ir.subgraph_input. Defaults to an empty dictionary.

    Returns:
        None: If #subgraph has no output tensors.
        Tensor:
            The output tensor of the call in the parent graph, if #subgraph has
            exactly 1 output.
        Tuple[Tensor]:
            Tuple of the output tensors of the call in the parent graph, if
            #subgraph has >1 outputs. The tensors will be in ascending order of
            the graph output index of the corresponding subgraph tensor.
    """
    subgraph_in_to_parent_in = subgraph_in_to_parent_in if subgraph_in_to_parent_in is not None else {}

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    pb_sg = subgraph._pb_graph

    op_name = g.name + '--call--' + subgraph.name

    opid = _ir.OperatorIdentifier("ai.graphcore", "Call", 1, _ir.NumInputs(),
                                  0)

    pb_callop = pb_g.createOp_CallOp(opid, subgraph._pb_graph,
                                     ctx._get_op_settings(op_name))

    # 1. Connect explictly passed inputs. These would have been created first
    #    by ir.get_graph, so we do them first. ir.get_graph will have created
    #    the input tensors t_0,...,t_N at input indices 0,..,N, respectively. We
    #    require that the user has passed the parent tensors that correspond to
    #    these inputs in the exact same order, so we can trivially reconstruct
    #    the input indices here.
    sgInIdx = 0
    for t in subgraph_fn_param_inputs:
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

    outnames = []

    def id_like_subgraph_tensor(tensor_id: str) -> str:
        return g._create_tensor_id(
            _ir.addScope(pb_g, _ir.removeScope(pb_sg, tensor_id)))

    for pb_sg_out_id in pb_sg.getOutputIds():
        sgOutIdx = pb_sg.getOutputIndex(pb_sg_out_id)
        # callOutIdx = pb_callop.subgraphOutToOpOutIndex(sgOutIdx)
        callOutIdx = sgOutIdx
        parent_tensor_id = id_like_subgraph_tensor(pb_sg_out_id)
        pb_callop.createAndConnectOutTensor(callOutIdx, parent_tensor_id)

        outnames.append(parent_tensor_id)

    out_tensors = [
        Tensor._from_pb_tensor(pb_g.getTensor(out)) for out in outnames
    ]

    pb_callop.setup()

    # Return nothing if no outputs.
    if len(out_tensors) == 0:
        return
    # Return single tensor if only one output.
    if len(out_tensors) == 1:
        return out_tensors[0]
    # Return tuple of output tensors if multiple outputs.
    else:
        return tuple(out_tensors)
