# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl


def test_tensor_id_conflict():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        name0 = popxl.variable(1, name="tensor").id
        name1 = popxl.variable(1, name="tensor").id
        name2 = popxl.constant(1, name="tensor").id
    assert name0 == "tensor"
    ids = [name0, name1, name2]
    assert len(ids) == len(set(ids))


def test_tensor_id_conflict_between_Ir():
    ir1 = popxl.Ir()
    with ir1.main_graph:
        t1 = popxl.variable(1, dtype=popxl.float32, name="tensor")

    ir2 = popxl.Ir()
    with ir2.main_graph:
        t2 = popxl.variable(1, dtype=popxl.float32, name="tensor")

    assert 2 == len(set([t1, t2]))  # test __hash__
    assert t1 != t2  # test __eq__


def test_basic():
    ir = popxl.Ir()
    g = ir.main_graph
    assert isinstance(g, popxl.Graph)
    str(g)  # test __repr__


def test_has_name():
    ir = popxl.Ir()
    g = ir.main_graph
    name = g.name
    assert isinstance(name, str)
    hash(g.id)  # test __hash__


def test_graph_cache():
    ir = popxl.Ir()
    pb_g = ir.main_graph._pb_graph
    g1 = popxl.Graph._from_pb(pb_g)
    g2 = popxl.Graph._from_pb(pb_g)
    assert g1 is g2  # test Ir._graph_cache
