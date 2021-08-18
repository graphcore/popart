# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

import popart
import popart._internal.ir as _ir


def check_graph_inputs(graph, graph_inputs):
    """ Helper function, graph inputs are what we expect. """
    assert graph.getInputIds() == graph_inputs, \
        f"Expected Graph {graph.getGraphString()} to have inputs {graph_inputs}, got {graph.getInputIds()}"


def check_graph_outputs(graph, graph_outputs):
    """ Helper function, graph outputs are what we expect. """
    assert graph.getOutputIds() == graph_outputs, \
        f"Expected Graph {graph.getGraphString()} to have outputs {graph_outputs}, got {graph.getOutputIds()}"


def test_graph_construction():
    """ Test that we can construct a popart._internal.ir.Graph object. """
    ir = _ir.Ir()
    g1Id = _ir.GraphId("g1")
    g1 = _ir.Graph(ir, g1Id)


def test_graph_graph_inputs():
    """ Test we can add/remove graph inputs. """
    ir = _ir.Ir()
    g1Id = _ir.GraphId("g1")
    g1 = ir.createGraph(g1Id)

    # Check initially graph inputs are empty.
    check_graph_inputs(g1, [])

    # Check addInput.
    g1.addInput(popart.TensorId("inputA"),
                _ir.TensorInfo(_ir.DataType.FLOAT16, [5, 5]))
    check_graph_inputs(g1, ["inputA"])
    g1.addInput(popart.TensorId("inputB"),
                _ir.TensorInfo(_ir.DataType.FLOAT, [65, 5]))
    check_graph_inputs(g1, ["inputA", "inputB"])
    g1.addInput(1, popart.TensorId("input1"),
                _ir.TensorInfo(_ir.DataType.FLOAT, [65, 5]), False)
    check_graph_inputs(g1, ["inputA", "input1", "inputB"])
    g1.addInput(1, popart.TensorId("input2"),
                _ir.TensorInfo(_ir.DataType.FLOAT, [65, 5]), True)
    check_graph_inputs(g1, ["inputA", "input2", "inputB"])

    # Check getInputId.
    assert g1.getInputId(0) == "inputA"
    assert g1.getInputId(1) == "input2"
    assert g1.getInputId(2) == "inputB"

    # Check getInputIndex
    assert g1.getInputIndex(popart.TensorId("inputA")) == 0
    assert g1.getInputIndex(popart.TensorId("input2")) == 1
    assert g1.getInputIndex(popart.TensorId("inputB")) == 2
    with pytest.raises(popart.popart_exception) as excinfo:
        g1.getInputIndex(popart.TensorId("nonExistingTensor"))

    # Check hasInputId.
    assert g1.hasInputId(popart.TensorId("inputA"))
    assert not g1.hasInputId(popart.TensorId("input1"))

    # Check removeInput.
    g1.removeInput(1)
    check_graph_inputs(g1, ["inputA", "inputB"])
    g1.removeInput(popart.TensorId("inputA"))
    check_graph_inputs(g1, ["inputB"])


def test_graph_graph_outputs():
    """ Test we can add/remove graph outputs. """
    ir = _ir.Ir()
    g1Id = _ir.GraphId("g1")
    g1 = ir.createGraph(g1Id)

    # We add inputs as a way of adding tensors to the graph that we can mark as
    # outputs.
    g1.addInput(popart.TensorId("t0"),
                _ir.TensorInfo(_ir.DataType.FLOAT16, [5, 5]))
    g1.addInput(popart.TensorId("t1"),
                _ir.TensorInfo(_ir.DataType.FLOAT16, [5, 5]))
    g1.addInput(popart.TensorId("t2"),
                _ir.TensorInfo(_ir.DataType.FLOAT16, [5, 5]))

    # Check markAsInput.
    check_graph_outputs(g1, [])
    g1.markAsOutput(popart.TensorId("t0"))
    check_graph_outputs(g1, ["t0"])
    g1.markAsOutput(0, popart.TensorId("t1"), False)
    check_graph_outputs(g1, ["t1", "t0"])
    g1.markAsOutput(0, popart.TensorId("t2"), True)
    check_graph_outputs(g1, ["t2", "t0"])

    # Check getOutputId.
    assert g1.getOutputId(0) == "t2"
    assert g1.getOutputId(1) == "t0"

    # Check getOutputIndex
    assert g1.getOutputIndex(popart.TensorId("t2")) == 0
    assert g1.getOutputIndex(popart.TensorId("t0")) == 1
    with pytest.raises(popart.popart_exception) as excinfo:
        g1.getOutputIndex(popart.TensorId("nonExistingTensor"))

    # Check hasOutputId.
    assert g1.hasOutputId(popart.TensorId("t0"))
    assert g1.hasOutputId(popart.TensorId("t2"))
    assert not g1.hasOutputId(popart.TensorId("t1"))

    # Check removeInput.
    g1.removeOutput(1)
    check_graph_outputs(g1, ["t2"])
    g1.removeOutput(popart.TensorId("t2"))
    check_graph_outputs(g1, [])


def test_graph_scope_functions():
    """ Test we can scope functions. """
    ir = _ir.Ir()
    g1Id = _ir.GraphId("g1")
    g1 = ir.createGraph(g1Id)

    # Test addScope
    assert g1.addScope(popart.TensorId("tensor1")) == "g1/tensor1"
    assert g1.addScope(popart.TensorId("foobar")) == "g1/foobar"

    # Test removeScope
    assert g1.removeScope(popart.TensorId("g1/tensor1")) == "tensor1"
    assert g1.removeScope(popart.TensorId("g1/foobar")) == "foobar"

    with pytest.raises(popart.popart_exception) as excinfo:
        g1.removeScope(popart.TensorId("h1/tensor1"))

    # Test getScope
    assert g1.getScope().str() == "g1"


def test_graph_id_member():
    """ Test .id member binding. """
    ir = _ir.Ir()
    g1Id = _ir.GraphId("g1")
    g1 = ir.createGraph(g1Id)

    assert g1.id == g1Id


def test_graph_get_graph_string():
    """ Test getGraphString binding. """
    ir = _ir.Ir()
    g1Id = _ir.GraphId("g1")
    g1 = ir.createGraph(g1Id)

    assert ir.getMainGraph().getGraphString() == "the main graph"
    assert g1.getGraphString() == "subgraph 'g1'"
