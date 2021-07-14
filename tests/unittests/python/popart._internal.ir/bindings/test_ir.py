# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart._internal.ir as _ir


def check_has_graph(ir, graphId):
    """ Helper function, test an IR has a Graph with ID graphId. """
    assert ir.hasGraph(graphId), \
        f"Expected IR to have a Graph with GraphId '{graphId.str()}'"


def check_does_not_have_graph(ir, graphId):
    """ Helper function, test an IR has a Graph with ID graphId. """
    assert not ir.hasGraph(graphId), \
        f"Did not expect IR to have a Graph with GraphId '{graphId.str()}'"


def check_can_get_graph(ir, graphId):
    """ Helper function, test an IR has a Graph with ID graphId. """
    g = ir.getGraph(graphId)
    assert g.id == graphId, f"Expected Graph ID {graphId.str()}, got {g.id.str()}"


def check_cant_get_graph(ir, graphId):
    """ Helper function, test an IR has a Graph with ID graphId. """
    with pytest.raises(IndexError):
        ir.getGraph(graphId)


def check_existing_graphs(ir, graphIdStrs):
    """ Helper function, test list of Graph IDs in IR. """
    ids = [g.id.str() for g in ir.getAllGraphs()]
    assert ids.sort() == graphIdStrs.sort(), \
        f"Expected IR to have Graph IDs {graphIdStrs.sort()}, got {ids.sort()}"


def test_ir_construction():
    """ Test that we can construct a popart._internal.ir.Ir object. """
    _ = _ir.Ir()


def test_ir_graph_management0():
    """ Test that we can create / test for presence of graphs. """
    ir = _ir.Ir()
    a = _ir.GraphId("A")
    b = _ir.GraphId("B")

    check_does_not_have_graph(ir, a)
    check_does_not_have_graph(ir, b)
    _ = ir.createGraph(a)
    check_has_graph(ir, a)
    check_does_not_have_graph(ir, b)
    _ = ir.createGraph(b)
    check_has_graph(ir, a)
    check_has_graph(ir, b)


def test_ir_graph_management1():
    """ Test that we can remove graphs. """
    ir = _ir.Ir()
    a = _ir.GraphId("A")

    check_does_not_have_graph(ir, a)
    _ = ir.createGraph(a)
    check_has_graph(ir, a)
    _ = ir.removeGraph(a)
    check_does_not_have_graph(ir, a)


def test_ir_graph_management2():
    """ Test that we can / can't get graphs. """
    ir = _ir.Ir()
    a = _ir.GraphId("A")
    b = _ir.GraphId("B")

    check_cant_get_graph(ir, a)
    check_cant_get_graph(ir, b)
    _ = ir.createGraph(a)
    check_can_get_graph(ir, a)
    check_cant_get_graph(ir, b)
    _ = ir.createGraph(b)
    check_can_get_graph(ir, a)
    check_can_get_graph(ir, b)


def test_ir_graph_management3():
    """ Test we can get all graphs. """
    ir = _ir.Ir()
    a = _ir.GraphId("A")
    b = _ir.GraphId("B")

    check_existing_graphs(ir, [''])
    _ = ir.createGraph(a)
    check_existing_graphs(ir, ['', 'A'])
    _ = ir.createGraph(b)
    check_existing_graphs(ir, ['', 'A', 'B'])


def test_ir_graph_management4():
    """ Test we can get the main graph. """
    ir = _ir.Ir()
    a = _ir.GraphId("A")
    b = _ir.GraphId("B")

    mainGraph = ir.getMainGraph()
    assert ir.hasGraph(mainGraph.id)
