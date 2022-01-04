# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir


def test_tensor_id_conflict():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        name0 = pir.variable(1, name="tensor").id
        name1 = pir.variable(1, name="tensor").id
        name2 = pir.constant(1, name="tensor").id
    assert name0 == "tensor"
    ids = [name0, name1, name2]
    assert len(ids) == len(set(ids))


def test_tensor_id_conflict_between_Ir():
    ir1 = pir.Ir()
    with ir1.main_graph():
        t1 = pir.variable(1, dtype=pir.float32, name="tensor")

    ir2 = pir.Ir()
    with ir2.main_graph():
        t2 = pir.variable(1, dtype=pir.float32, name="tensor")

    assert 2 == len(set([t1, t2]))  # test __hash__
    assert t1 != t2  # test __eq__


def test_basic():
    ir = pir.Ir()
    g = ir.main_graph()
    assert isinstance(g, pir.Graph)
    str(g)  # test __repr__


def test_has_name():
    ir = pir.Ir()
    g = ir.main_graph()
    name = g.name
    assert isinstance(name, str)
    hash(g.id)  # test __hash__


def test_graph_cache():
    ir = pir.Ir()
    pb_g = ir.main_graph()._pb_graph
    g1 = pir.Graph._from_pb(pb_g)
    g2 = pir.Graph._from_pb(pb_g)
    assert g1 is g2  # test Ir._graph_cache
