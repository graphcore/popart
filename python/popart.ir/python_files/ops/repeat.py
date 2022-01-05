# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.graph import Graph
from popart.ir.tensor import Tensor

from .utils import check_in_graph

__all__ = ['repeat']


# TODO: T49287 add a repeat_with_info
@op_debug_context
def repeat(repeat_subgraph: Graph,
           repeat_trip_count: int,
           *subgraph_fn_param_inputs: Tensor,
           subgraph_in_to_parent_in: Optional[Mapping[Tensor, Tensor]] = None
           ) -> Union[None, Tensor, Tuple[Tensor, ...]]:
    """
    Repeat Op: An op that repeats a subgraph with the provided input tensors
        `repeat_trip_count` number of times..

    Implementation detail: In order to maintain the input / output indices of the subgraph, we must
        call the user provided subgraph, and create a "middle" subgraph to repeat the user provided
        subgraph inside:

    # yapf: disable

              LoopOp         Keep
                 |          Going     User
                 |  Iterator  |      Inputs
                 |     |      |      | | |
                 V     V      V      V V V
                 .-Wrapper_subgraph--+-+-+----------.
Parent graph     |                   | | |          |
                 |                   | | |          |
                 |                   | | |          |
                 |                   V V V          |
                 | CallOp .-Loop_subgraph---.       |
                 |    |   | (user provided) |       |
                 |    '-->|                 |       |
                 |        |     (Ops)       |       |
                 |        |                 |       |
                 |        |                 |       |
                 |        '----------+-+-+--'       |
                 |                   | | |          |
                 |                   V V V          |
                 '---+---------------+-+-+----------'
                     |               | | |
                     |               | | |
                     V               V V V
                    Keep            User outputs
                   Going

    # yapf: enable

    Args:
        repeat_subgraph (Graph): User defined graph to repeat `repeat_trip_count` times.
        repeat_trip_count (int): Number of times to repeat the subgraph.
        subgraph_in_to_parent_in (Optional[Mapping[Tensor, Tensor]]):
            Mapping of `subgraph tensor -> parent tensor` that corresponds to
            the inputs that the callable defined internally, e.g. by using
            popart.ir.subgraph_input. Defaults to an empty dictionary.
            Works effectively the same as the call op's `subgraph_in_to_parent_in` argument.

    Throws:
        ValueError: If repeat_trip_count <= 1.
        ValueError: If the number of explicitly passed inputs + the number of loop created inputs
            != the number of outputs.

    Returns:
        None: If `subgraph` has no output tensors.
        Tensor:
            The output tensor of the call in the parent graph, if `subgraph` has
            exactly 1 output.
        Tuple[Tensor, ...]:
            Tuple of the output tensors of the call in the parent graph, if
            #subgraph has >1 outputs. The tensors will be in ascending order of
            the graph output index of the corresponding subgraph tensor.


    Example:
    ```
        # popart.ir.Module to repeat
        class AddWeight(pir.Module):
        def __init__(self):
            self.w: pir.Tensor = None

        def build(self, x):
            self.w = pir.subgraph_input(x.shape, x.dtype, "w")
            return self.w + x, w

        with g: # a graph
            add_weight0 = AddWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x0)

            # repeat 8 times
            y0, w0 = ops.repeat(add_weight_graph0,
                                8,
                                x0,
                                subgraph_in_to_parent_in={add_weight0.w: w0})
    ```
    """

    if repeat_trip_count <= 1:
        raise ValueError(
            f"Repeat trip count for repeat of {repeat_subgraph.name} "
            f"of {repeat_trip_count} must be > 1.")

    subgraph_in_to_parent_in = subgraph_in_to_parent_in if (
        subgraph_in_to_parent_in is not None) else {}

    # For clarity, we rename our graphs:
    # - Bottom: The user provided bottom level graph. We call this with a call op. This has gone
    #       through the create_graph procedure, so we do not need to add subgraph ins/outs.
    # - Middle: The graph we create to wrap the bottom graph. We repeat this. This has not gone
    #       through the create_graph procedure, so we can add subgraph inputs (including the repeat
    #       iterator and condition) as needed.
    # - Top: The graph we add the repeat to, and the current graph in the context. Potentially
    #       can be the main graph.
    ctx = get_current_context()
    top_graph = ctx.graph
    ir = top_graph.ir()
    pb_ir = ir._pb_ir
    pb_top_graph = top_graph._pb_graph
    # This is the graph we will call.
    bottom_graph = repeat_subgraph
    pb_bottom_graph = bottom_graph._pb_graph

    # The loop op requires the same number of inputs as outputs.
    if len(subgraph_fn_param_inputs) + len(subgraph_in_to_parent_in) != len(
            pb_bottom_graph.getOutputIds()):
        raise ValueError(
            f"The number of explicitly passed inputs ({len(subgraph_fn_param_inputs)}):"
            f" {[t.id for t in subgraph_fn_param_inputs]}\n"
            f" + the number of loop created inputs ({len(subgraph_in_to_parent_in)}):"
            f" {[t.id for t in subgraph_in_to_parent_in.values()]}\n"
            f" must equal the number of outputs ({len(pb_bottom_graph.getOutputIds())}):"
            f" {pb_bottom_graph.getOutputIds()}")

    # Create the middle graph, call and loop ops
    pb_middle_graph, pb_callop, pb_loop_op = _setup_call_and_repeat(
        pb_ir, pb_top_graph, pb_bottom_graph)

    # set the number of times to loop
    pb_loop_op.setTripCountValue(repeat_trip_count)
    # Check all the parent tensors are in the right graph.
    for _, parent_tensor in subgraph_in_to_parent_in.items():
        check_in_graph(top_graph, parent_tensor)

    # 1, 2. Connect inputs.
    _setup_inputs(subgraph_fn_param_inputs, subgraph_in_to_parent_in,
                  pb_top_graph, pb_bottom_graph, pb_middle_graph, pb_callop,
                  pb_loop_op)

    # 3. Connect outputs.
    outnames = _setup_outputs(pb_top_graph, pb_bottom_graph, pb_middle_graph,
                              pb_callop, pb_loop_op)

    pb_callop.setup()
    pb_loop_op.setup()

    out_tensors = [
        Tensor._from_pb_tensor(pb_top_graph.getTensor(out)) for out in outnames
    ]

    # Return nothing if no outputs.
    if len(out_tensors) == 0:
        return None
    # Return single tensor if only one output.
    if len(out_tensors) == 1:
        return out_tensors[0]
    # Return tuple of output tensors if multiple outputs.
    else:
        return tuple(out_tensors)


# Design point: For simplicity all of the below functions only take _ir level objects as arguments.
def _setup_call_and_repeat(pb_ir: _ir.Ir, pb_top_graph: _ir.Graph,
                           pb_bottom_graph: _ir.Graph
                           ) -> Tuple[_ir.Graph, _ir.op.CallOp, _ir.op.LoopOp]:
    """Setup the call and repeat ops, as well as the middle graph that the loop op will loop.

    Args:
        pb_ir (_ir.Ir): The _ir level Ir
        pb_top_graph (_ir.Graph): The _ir top level graph that will contain the loop op.
        pb_bottom_graph (_ir.Graph): The _ir user defined subgraph that will be called.

    Returns:
        Tuple[_ir.Graph, _ir.op.CallOp, _ir.op.LoopOp]: The created _ir-level middle graph, call op
            and loop op.
    """
    # This is the graph we will repeat.
    pb_middle_graph = pb_ir.createGraph(
        _ir.GraphId(
            pb_ir.createUniqueSubgraphId(
                f"{pb_bottom_graph.id.str()}__loop_wrapper")))

    opid = _ir.OperatorIdentifier("ai.graphcore", "Call", 1, _ir.NumInputs(),
                                  0)
    op_name = pb_middle_graph.id.str() + '__call__' + pb_bottom_graph.id.str()

    ctx = get_current_context()
    # Call the bottom_graph
    pb_callop = pb_middle_graph.createOp_CallOp(opid, pb_bottom_graph,
                                                ctx._get_op_settings(op_name))

    opid = _ir.OperatorIdentifier("ai.onnx", "Loop", 11, _ir.NumInputs(), 0)
    op_name = pb_top_graph.id.str() + '__loop__' + pb_middle_graph.id.str()

    # Loop the middle_graph
    pb_loop_op = pb_top_graph.createOp_LoopOp(opid,
                                              ctx._get_op_settings(op_name),
                                              pb_middle_graph)

    # Add mandatory loop iterator tensor to subgraph (is not an output)
    repeatIterId = _ir.addScope(pb_middle_graph, "Iterator___")
    pb_middle_graph.addInput(repeatIterId,
                             _ir.TensorInfo(_ir.DataType.INT32, ()))

    # Add mandatory loop condition tensor to subgraph (is also an output)
    repeatCondId = _ir.addScope(pb_middle_graph, "LoopCond___")
    pb_middle_graph.addInput(repeatCondId, _ir.TensorInfo(
        _ir.DataType.BOOL, ()))
    pb_middle_graph.markAsOutput(repeatCondId)

    return pb_middle_graph, pb_callop, pb_loop_op


def _setup_inputs(subgraph_fn_param_inputs: Iterable[Tensor],
                  subgraph_in_to_parent_in: Mapping[Tensor, Tensor],
                  pb_top_graph: _ir.Graph, pb_bottom_graph: _ir.Graph,
                  pb_middle_graph: _ir.Graph, pb_callop: _ir.op.CallOp,
                  pb_loop_op: _ir.op.LoopOp) -> None:
    """Do the following:

    1. Connect explicitly passed inputs. These would have been created first
       by ir.get_graph, so we do them first. ir.get_graph will have created
       the input tensors t_0,...,t_N at input indices 0,..,N, respectively. We
       require that the user has passed the parent tensors that correspond to
       these inputs in the exact same order, so we can trivially reconstruct
       the input indices here.

    2. Connect internally created inputs.

    Args:
        subgraph_fn_param_inputs (Iterable[Tensor]): User defined explicit inputs.
        subgraph_in_to_parent_in (Mapping[Tensor, Tensor]):
            Mapping of `subgraph tensor -> parent tensor` that corresponds to
            the inputs that the callable defined internally, e.g. by using
            popart.ir.subgraph_input. Defaults to an empty dictionary.
            Works effectively the same as the call op's `subgraph_in_to_parent_in` argument.
        pb_top_graph (_ir.Graph): Top _ir graph
        pb_bottom_graph (_ir.Graph): Bottom _ir Graph
        pb_middle_graph (_ir.Graph): Middle _ir Graph
        pb_callop (_ir.op.CallOp): Previously created Call op
        pb_loop_op (_ir.op.LoopOp): Previously created Loop op
    """

    # Note: Only bottom_graph (which is called) has gone through the ir.get_graph process.
    # middle_graph (intentionally) has not, so we need to add loop inputs/outputs.
    # User defined indices start at 2 for loop ops.
    sgInIdx = 0
    for t in subgraph_fn_param_inputs:
        callInIdx = pb_callop.subgraphInToOpInIndex(sgInIdx)
        # Note the + 2 here
        pb_loop_op.addLoopInput(sgInIdx + 2,
                                _ir.addScope(pb_top_graph, t.name),
                                _ir.addScope(pb_middle_graph, t.name), False)
        pb_callop.connectInTensor(callInIdx,
                                  _ir.addScope(pb_middle_graph, t.name))
        sgInIdx += 1

    # 2. Connect internally created inputs.
    for sg_tensor, parent_tensor in subgraph_in_to_parent_in.items():
        sgInIdx = pb_bottom_graph.getInputIndex(sg_tensor.id)
        callInIdx = pb_callop.subgraphInToOpInIndex(sgInIdx)

        top_tensor_id = _ir.addScope(pb_top_graph, parent_tensor.id)
        pb_loop_op.addLoopInput(
            sgInIdx + 2, top_tensor_id,
            _ir.addScope(pb_middle_graph,
                         _ir.removeScope(pb_bottom_graph, sg_tensor.id)),
            False)
        set_input_modified(pb_loop_op, pb_loop_op.inTensor(sgInIdx + 2))
        pb_callop.connectInTensor(
            callInIdx,
            _ir.addScope(pb_middle_graph,
                         _ir.removeScope(pb_bottom_graph, sg_tensor.id)))
        set_input_modified(pb_callop, pb_callop.inTensor(callInIdx))


def _setup_outputs(pb_top_graph: _ir.Graph, pb_bottom_graph: _ir.Graph,
                   pb_middle_graph: _ir.Graph, pb_callop: _ir.op.CallOp,
                   pb_loop_op: _ir.op.LoopOp) -> List[str]:
    """3. Connect outputs. We introspect the subgraph to get its outputs then,
         for each one, create an output tensor of the call op in the middle
         graph.
    Args:
        pb_top_graph (_ir.Graph): Top _ir graph
        pb_bottom_graph (_ir.Graph): Bottom _ir Graph
        pb_middle_graph (_ir.Graph): Middle _ir Graph
        pb_callop (_ir.op.CallOp): Previously created Call op
        pb_loop_op (_ir.op.LoopOp): Previously created Loop op

    Returns:
        List[str]: The output tensor ids.
    """

    outnames: List[str] = []

    for pb_repeat_subgraph_out_id in pb_bottom_graph.getOutputIds():
        top_tensor_id = _ir.addScope(
            pb_top_graph,
            _ir.removeScope(pb_bottom_graph, pb_repeat_subgraph_out_id))
        # Already has scope added
        middle_tensor_id = _ir.removeScope(pb_bottom_graph,
                                           pb_repeat_subgraph_out_id)
        bottom_tensor_id = _ir.addScope(
            pb_bottom_graph,
            _ir.removeScope(pb_bottom_graph, pb_repeat_subgraph_out_id))

        sgOutIdx = pb_bottom_graph.getOutputIndex(bottom_tensor_id)
        callOutIdx = pb_callop.subgraphOutToOpOutIndex(sgOutIdx)

        # Avoid tensor name collisions
        middle_tensor_id = pb_middle_graph.getIr().createIntermediateTensorId(
            middle_tensor_id)
        pb_callop.createAndConnectOutTensor(callOutIdx, middle_tensor_id)

        pb_middle_graph.markAsOutput(middle_tensor_id)
        sgOutIdx = pb_middle_graph.getOutputIndex(middle_tensor_id)
        repeatOutIdx = pb_loop_op.subgraphOutToOpOutIndex(sgOutIdx)
        # Avoid tensor name collisions
        top_tensor_id = pb_middle_graph.getIr().createIntermediateTensorId(
            top_tensor_id)
        # We overwrite here as we added the middle_tensor_id as an output above, but we want to make
        # sure the loop op is setup correctly.
        pb_loop_op.addLoopOutput(repeatOutIdx, top_tensor_id, middle_tensor_id,
                                 True)

        outnames.append(top_tensor_id)
    return outnames


# TODO: T49287 add a repeat_with_info: move/modify this function to match call op.
def set_input_modified(_op: _ir.Op, in_tensor: _ir.Tensor):
    """Specify that the input tensor `in_tensor` is modified by the repeat op.
        this will guarantee that any modification to the graph input during the execution
        of the called graph will also change `in_tensor`.
        The regions modified by the call op will be specified by the Ops in the called graph.

    Args:
        in_tensor (_ir.Tensor): Tensor to be modified.
    """
    index = _op.inIndex(in_tensor)
    _graph = _op.getCalledGraph()
    _sg_tensor = _graph.getInputTensor(_op.opInToSubgraphInIndex(index))
    _regions = _sg_tensor.modifiedRegionsByOps(_graph.getOps())
    _op.addModified(index, _regions)
