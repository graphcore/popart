# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart.ir as pir


def test_context_manager():
    ir = pir.Ir()
    main = ir.main_graph()

    exp_prefix = 'Trying to access a graph, but no graph has been selected.'

    with pytest.raises(RuntimeError) as excinfo:
        pir.gcg()
    assert str(excinfo.value).startswith(exp_prefix)

    with main:
        assert pir.gcg() == main

    with pytest.raises(RuntimeError) as excinfo:
        pir.gcg()
    assert str(excinfo.value).startswith(exp_prefix)


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


def test_basic():
    ir = pir.Ir()
    g = ir.main_graph()
    assert isinstance(g, pir.Graph)


def test_has_name():
    ir = pir.Ir()
    g = ir.main_graph()
    name = g.name
    assert isinstance(name, str)
