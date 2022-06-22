# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Iterable, List, Mapping, Optional, Tuple, Union, Dict

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.graph import Graph
from popxl.ops.call import CallSiteInfo
from popxl.tensor import Tensor

from .call import _prep_inputs_pre_validation, _validate_inputs, _prep_inputs, _validate_connected_tensors


@op_debug_context
def repeat(graph: Graph,
           repeat_count: int,
           *inputs: Union[Tensor, Iterable[Tensor]],
           inputs_dict: Optional[Mapping[Tensor, Tensor]] = None
           ) -> Tuple[Tensor, ...]:
    """
    Repeatedly call a graph.

    This operation repeatedly executes a graph `repeat_count` times. The
    input tensors are provided as graph inputs for the first iteration.

    The `inputs` and `inputs_dict` tensors are passed as graph inputs.
    You can specify an input either positionally using `inputs`, or via a tensor
    map using `inputs_dict`.

    Graph inputs are determined when the graph is created using `create_graph(callable, ...)`.
    The order of inputs will be the same as the order of the tensor inputs in the
    function signature and the order of called `popxl.graph_inputs`.
    See :py:meth:`~popxl.Ir.create_graph` for more information.

    Between each execution of the subgraph, the N outputs of subgraph will be copied to the first N inputs.
    These are called *loop carried* inputs. The number of outputs must be less than or equal to the number of inputs.
    The remaining inputs will be unchanged throughout the loop iterations (unless modified in place).

    Example:

    .. code-block:: python

        # popxl.Module to repeat
        class AddWeight(popxl.Module):
        def __init__(self):
            self.w: popxl.Tensor = None

        def build(self, x):
            self.w = popxl.graph_input(x.shape, x.dtype, "w")
            return self.w + x, w

        with g: # a graph
            add_weight0 = AddWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x0)

            # repeat 8 times
            y0, w0 = ops.repeat(add_weight_graph0,
                                8,
                                x0,
                                inputs_dict={add_weight0.w: w0})

    See also `PyTorch Tensor.repeat <https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html>`__, `NumPy repeat <https://numpy.org/doc/stable/reference/generated/numpy.repeat.html>`__.

    Args:
        graph (Graph): User defined graph to repeat `repeat_count` times.
        repeat_count (int): Number of times to repeat calling the graph.
        *inputs (Tensor, List[Tensor], int, float):
            Provide inputs via position.
        inputs_dict (Optional[Mapping[Tensor, Tensor]]):
            Provide inputs via a tensor map. Mapping of `graph tensor -> parent tensor`.
        check_inputs (bool = True):
            If true, then check when called that all inputs have been provided.

    Throws:
        ValueError: If repeat_count < 0.
        ValueError: If the number of subgraph inputs < subgraph outputs.

    Returns:
        Tuple[Tensor, ...]:
            Tuple of the output tensors of the call in the parent graph.
    """
    loop_info = repeat_with_info(graph,
                                 repeat_count,
                                 *inputs,
                                 inputs_dict=inputs_dict)

    out_tensors = loop_info.outputs
    return out_tensors


def repeat_with_info(graph: Graph,
                     repeat_count: int,
                     *inputs: Union[Tensor, Iterable[Tensor]],
                     inputs_dict: Optional[Mapping[Tensor, Tensor]] = None,
                     check_inputs: bool = True) -> CallSiteInfo:
    """
    Repeatedly call a graph and return information about the call site.

    This operation repeatedly executes a graph `repeat_count` number of times. The
    input tensors are provided as graph inputs for the first iteration.

    Returns `CallSiteInfo` that can be used to inspect callsite inputs/outputs.

    The `inputs` and `inputs_dict` tensors are passed as graph inputs.
    You can specify an input either positionally using `inputs` or via a tensor
    map using `inputs_dict`.

    Graph inputs are determined when the graph is created using `ir.create_graph(callable, ...)`.
    The order of inputs will be the same as the order of the tensor inputs in the
    function signature and the order of called `popxl.graph_inputs`.
    See :py:meth:`~popxl.Ir.create_graph` for more information.

    Between each execution of the subgraph, the N outputs of subgraph will be copied to the first N inputs.
    These are called *loop carried* inputs. The number of outputs must be less than or equal to the number of inputs.

    Implementation detail: In order to maintain the input / output indices of the subgraph, we must
    call the user provided subgraph, and create a "middle" subgraph to repeat the user provided
    subgraph inside:

    .. code-block:: none

                    LoopOp         Keep
                        |          Going  Loop Carried
                        |  Iterator  |      Inputs
                        |     |      |      | | |    |- Implicit Inputs
                        V     V      V      V V V    V
                        .-Wrapper_subgraph--+-+-+----+-----.
        Parent graph    |                   | | |    |     |
                        |                   | | |    |     |
                        |                   | | |    |     |
                        |                   V V V    |     |
                        | CallOp .-Loop_subgraph---. |     |
                        |    |   | (user provided) |<-     |
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
                            Keep         Loop Carried
                        Going              Outputs
    Example:

    .. code-block:: python

        # popxl.Module to repeat
        class AddWeight(popxl.Module):
        def __init__(self):
            self.w: popxl.Tensor = None

        def build(self, x):
            self.w = popxl.graph_input(x.shape, x.dtype, "w")
            return self.w + x, w

        with g: # a graph
            add_weight0 = AddWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x0)

            # repeat 8 times
            call_info = ops.repeat(add_weight_graph0,
                                   8,
                                   x0,
                                   inputs_dict={add_weight0.w: w0})
            y0, w0 = call_info.outputs()

    Args:
        graph (Graph): User defined graph to repeat `repeat_count` times.
        repeat_count (int): Number of times to repeat calling the graph.
        *inputs (Tensor, List[Tensor], int, float):
            Provide inputs via position.
        inputs_dict (Optional[Mapping[Tensor, Tensor]]):
            Provide inputs via a tensor map. Mapping of `graph tensor -> parent tensor`.
        check_inputs (bool):
            Check when called if all inputs have been provided. Defaults to True.

    Raises:
        ValueError: If repeat_count < 0.
        ValueError: If the number of explicitly passed inputs + the number of loop created inputs
            != the number of outputs.

    Returns:
        repeat_info (CallSiteInfo): Information on the created callsite for
        the repeat op.
    """

    # Validate inputs
    if repeat_count < 0:
        raise ValueError(
            f"Repeat count must be >= 0. Repeat count: {repeat_count}. Graph: {graph}"
        )

    # The loop op requires the same number of inputs as outputs.
    total_inputs = len(graph.inputs)
    total_outputs = len(graph.outputs)
    if total_inputs < total_outputs:
        raise ValueError(
            f"To repeat the subgraph ({graph.id}) the number of inputs must be greater than or equal to the number of outputs."
            f" {total_inputs} < {total_outputs}")

    # For clarity, we rename our graphs:
    # - Bottom: The user provided bottom level graph. We call this with a call op. This has gone
    #       through the create_graph procedure, so we do not need to add graph ins/outs.
    # - Middle: The graph we create to wrap the bottom graph. We repeat this. This has not gone
    #       through the create_graph procedure, so we can add graph inputs (including the repeat
    #       iterator and condition) as needed.
    # - Top: The graph we add the repeat to, and the current graph in the context. Potentially
    #       can be the main graph.
    ctx = get_current_context()
    top_graph = ctx.graph
    ir = top_graph.ir
    pb_ir = ir._pb_ir
    pb_top_graph = top_graph._pb_graph
    # This is the graph we will call.
    bottom_graph = graph
    pb_bottom_graph = bottom_graph._pb_graph

    # Create the middle graph, call and loop ops
    pb_middle_graph, pb_callop, pb_loop_op = _setup_call_and_repeat(
        pb_ir, pb_top_graph, pb_bottom_graph)

    # set the number of times to loop
    pb_loop_op.setTripCountValue(repeat_count)

    # Prep and validate inputs
    inputs, inputs_dict = _prep_inputs_pre_validation(inputs, inputs_dict)
    _validate_inputs(top_graph, graph, inputs, inputs_dict)
    inputs, inputs_dict = _prep_inputs(graph, inputs, inputs_dict)

    # 1, 2. Connect inputs.
    connected_inputs = _setup_inputs(
        inputs=inputs,
        inputs_dict=inputs_dict,
        pb_top_graph=pb_top_graph,
        pb_bottom_graph=pb_bottom_graph,
        pb_middle_graph=pb_middle_graph,
        pb_callop=pb_callop,
        pb_loop_op=pb_loop_op,
    )

    _validate_connected_tensors(check_inputs, connected_inputs, graph, inputs,
                                inputs_dict)

    # 3. Connect outputs.
    _ = _setup_outputs(pb_top_graph, pb_bottom_graph, pb_middle_graph,
                       pb_callop, pb_loop_op)

    pb_callop.setup()
    pb_loop_op.setup()

    r_info = CallSiteInfo(pb_loop_op)  # repeat info
    c_info = CallSiteInfo(pb_callop)  # call info

    # Modified tensors for the called graph (bottom)
    loop_carried_inputs = pb_loop_op.getNumExplicitInputs()
    for bottom_t in bottom_graph._by_ref_inputs:
        middle_t = c_info.graph_to_parent(bottom_t)
        loop_carried = pb_middle_graph.getInputIndex(
            middle_t.id) < loop_carried_inputs
        # If a tensor was set as a by_ref_input, we should also do the same for the looped subgraph.
        c_info.set_parent_input_modified(
            middle_t, infer_modified_regions=not loop_carried)
        top_t = r_info.graph_to_parent(middle_t)
        r_info.set_parent_input_modified(
            top_t, infer_modified_regions=not loop_carried)
        r_info.called_graph._by_ref_inputs.add(middle_t)

    return r_info


# Design point: For simplicity all of the below functions only take _ir level objects as arguments.
def _setup_call_and_repeat(pb_ir: _ir.Ir, pb_top_graph: _ir.Graph,
                           pb_bottom_graph: _ir.Graph
                           ) -> Tuple[_ir.Graph, _ir.op.CallOp, _ir.op.LoopOp]:
    """Set up the call and repeat ops, as well as the middle graph that the loop op will loop over.

    Args:
        pb_ir (_ir.Ir): The _ir level Ir
        pb_top_graph (_ir.Graph): The top-level graph in `_ir` that will contain the loop op.
        pb_bottom_graph (_ir.Graph): The user-defined graph in `_ir` that will be called.

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

    # Add mandatory loop iterator tensor to graph (is not an output)
    repeatIterId = _ir.addScope(pb_middle_graph, "Iterator___")
    pb_middle_graph.addInput(repeatIterId,
                             _ir.TensorInfo(_ir.DataType.INT32, ()))

    # Add mandatory loop condition tensor to graph (is also an output)
    repeatCondId = _ir.addScope(pb_middle_graph, "LoopCond___")
    pb_middle_graph.addInput(repeatCondId, _ir.TensorInfo(
        _ir.DataType.BOOL, ()))
    pb_middle_graph.markAsOutput(repeatCondId)

    return pb_middle_graph, pb_callop, pb_loop_op


def _setup_inputs(inputs: Iterable[Tensor],
                  inputs_dict: Mapping[Tensor, Tensor],
                  pb_top_graph: _ir.Graph, pb_bottom_graph: _ir.Graph,
                  pb_middle_graph: _ir.Graph, pb_callop: _ir.op.CallOp,
                  pb_loop_op: _ir.op.LoopOp) -> None:
    """Set up the inputs.

    This is done in the following way:

    1. Connect explicitly passed inputs. These would have been created first
       by :py:meth:`~popxl.Ir.create_graph`, so we do them first. ``create_graph()`` will have created
       the input tensors `t_0`, ..., `t_N` at input indices 0, ..., N, respectively.
       The parent tensors that correspond to
       these inputs must be passed in the exact same order, so we can trivially reconstruct
       the input indices here.

    2. Connect internally created inputs.

    Args:
        inputs (Iterable[Tensor]): User defined explicit inputs.
        inputs_dict (Mapping[Tensor, Tensor]):
            Mapping of `graph tensor -> parent tensor` that corresponds to
            the inputs that the callable defined internally, for example by using
            popxl.graph_input. Defaults to an empty dictionary.
            Works effectively the same as the `call` op's `inputs_dict` argument.
        pb_top_graph (_ir.Graph): Top _ir graph.
        pb_bottom_graph (_ir.Graph): Bottom _ir graph.
        pb_middle_graph (_ir.Graph): Middle _ir graph
        pb_callop (_ir.op.CallOp): Previously created call op.
        pb_loop_op (_ir.op.LoopOp): Previously created loop op.
    """

    # Note: Only bottom_graph (which is called) has gone through the ir.get_graph process.
    # middle_graph (intentionally) has not, so we need to add loop inputs/outputs.
    # User defined indices start at 2 for loop ops.
    connected_inputs: List[Dict] = []
    for graph_input_idx, parent_tensor in enumerate(inputs):
        call_in_idx = pb_callop.subgraphInToOpInIndex(graph_input_idx)
        # Note the + 2 here
        pb_loop_op.addLoopInput(
            call_in_idx + 2, _ir.addScope(pb_top_graph, parent_tensor.name),
            _ir.addScope(pb_middle_graph, parent_tensor.name), False)
        pb_callop.connectInTensor(
            call_in_idx, _ir.addScope(pb_middle_graph, parent_tensor.name))
        connected_inputs += [{
            'graph_input_idx': graph_input_idx,
            'parent_tensor': parent_tensor
        }]

    # 2. Connect internally created inputs.
    for graph_tensor, parent_tensor in inputs_dict.items():
        graph_input_idx = pb_bottom_graph.getInputIndex(graph_tensor.id)
        call_in_idx = pb_callop.subgraphInToOpInIndex(graph_input_idx)

        top_tensor_id = _ir.addScope(pb_top_graph, parent_tensor.id)
        pb_loop_op.addLoopInput(
            call_in_idx + 2, top_tensor_id,
            _ir.addScope(pb_middle_graph,
                         _ir.removeScope(pb_bottom_graph, graph_tensor.id)),
            False)
        pb_callop.connectInTensor(
            call_in_idx,
            _ir.addScope(pb_middle_graph,
                         _ir.removeScope(pb_bottom_graph, graph_tensor.id)))
        connected_inputs += [{
            'graph_input_idx': graph_input_idx,
            'parent_tensor': parent_tensor
        }]

    return connected_inputs


def _setup_outputs(pb_top_graph: _ir.Graph, pb_bottom_graph: _ir.Graph,
                   pb_middle_graph: _ir.Graph, pb_callop: _ir.op.CallOp,
                   pb_loop_op: _ir.op.LoopOp) -> List[str]:
    """Connect outputs (step 3).

    We introspect the graph to get its outputs then, for each one, create an output tensor of the
    call op in the middle graph.

    Args:
        pb_top_graph (_ir.Graph): Top _ir graph.
        pb_bottom_graph (_ir.Graph): Bottom _ir graph.
        pb_middle_graph (_ir.Graph): Middle _ir graph.
        pb_callop (_ir.op.CallOp): Previously created call op.
        pb_loop_op (_ir.op.LoopOp): Previously created loop op.

    Returns:
        List[str]: The output tensor IDs.
    """

    outnames: List[str] = []

    for pb_subgraph_out_id in pb_bottom_graph.getOutputIds():
        top_tensor_id = _ir.addScope(
            pb_top_graph, _ir.removeScope(pb_bottom_graph, pb_subgraph_out_id))
        # Already has scope added
        middle_tensor_id = _ir.removeScope(pb_bottom_graph, pb_subgraph_out_id)
        bottom_tensor_id = _ir.addScope(
            pb_bottom_graph,
            _ir.removeScope(pb_bottom_graph, pb_subgraph_out_id))

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
