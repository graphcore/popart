# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popart._internal.ir import Ir, Graph, GraphId, TensorInfo, DataType, Tensors, Scope, TensorType
import numpy as np


def test_tensors_construction():
    """ Test that we can construct a popart._internal.ir.Graph object. """
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    _ = Tensors(graph)


def test_get():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add some tensors.
    ids = [i for i in "abcdefghi"]
    for tid in ids:
        ts.addActGrad(tid)

    # Get the tensors one by one and confirm we have been returned the correct tensor.
    for tid in ids:
        t = ts.get(tid)
        assert t.id == tid


def test_contains_with_scope():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    ts.addActGrad('a/b/c/foo')
    ts.addActGrad('a/b/bar')

    scope = Scope() / 'a' / 'b' / 'c'

    assert ts.contains('foo', scope)
    assert ts.contains('bar', scope)
    assert not ts.contains('fizz', scope)


def test_find():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add three tensors called foo with different scopes.
    ts.addActGrad('foo')
    ts.addActGrad('a/foo')
    ts.addActGrad('a/b/c/foo')

    # Make sure we can find all three tensors.
    foo = ts.find('foo', Scope())
    assert foo == 'foo'
    foo = ts.find('foo', Scope() / 'a')
    assert foo == 'a/foo'
    foo = ts.find('foo', Scope() / 'a' / 'b' / 'c')
    assert foo == 'a/b/c/foo'


def test_adding_actGrads():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add some tensors.
    ids = [i for i in "abcdefghi"]
    for tid in ids:
        ts.addActGrad(tid)

    # Check the number of tensors is correct.
    assert ts.n() == len(ids)


def test_contains():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add some tensors.
    ids = [i for i in "abcdefghi"]
    for tid in ids:
        ts.addActGrad(tid)

    # Check all expected tensors are in ts.
    for tid in ids:
        assert ts.contains(tid)

    # Check `ts.contains` is not just returning true.
    for tid in 'xyz':
        assert not ts.contains(tid)


def test_getAllTensorIds():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add some tensors.
    ids = [i for i in "abcdefghi"]
    for tid in ids:
        ts.addActGrad(tid)

    # Check ids returned from getAllTensorIds.
    assert set(ts.getAllTensorIds()) == set(ids)


def test_remove():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add some tensors.
    ids = [i for i in "abcdefghi"]
    for tid in ids:
        ts.addActGrad(tid)

    # Test removing tensors
    while ids:
        x = ids[0]
        del ids[0]
        ts.remove(x)
        assert not ts.contains(x)
        assert ts.n() == len(ids)


def test_remove_all_isolated():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add some tensors.
    ids = [i for i in "abcdefghi"]
    for tid in ids:
        ts.addActGrad(tid)

    # All these tensors should be isolated
    ts.removeIsolated(False)
    assert ts.n() == 0


def test_add_var_init():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    tinfo = TensorInfo(DataType.FLOAT, data.shape)
    ts.addVarInit("data", tinfo, data)


def test_add_const_init():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    tinfo = TensorInfo(DataType.FLOAT, data.shape)
    ts.addConstInit("data", tinfo, data)


def test_add_stream():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    tinfo = TensorInfo(DataType.FLOAT, [10])
    ts.addStream("data", tinfo)


def test_make_const_init():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    # Add a tensor and check the value returned by `tensorType()`.
    ts.addActGrad('foo')
    t = ts.get('foo')
    assert t.tensorType() == TensorType.ActGrad

    # Make the tensor const init and check the value returned by `tensorType()` has changed.
    t.info = TensorInfo(DataType.FLOAT, data.shape)
    ts.makeConstInit('foo', data)
    assert t.tensorType() == TensorType.Const

    # TODO(T42205): Test that the tensor data matches the numpy array, `data`.
    # d = t.tensorData();


def test_get_ids():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    for tid in 'abcd':
        ts.addActGrad(tid)

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    tinfo = TensorInfo(DataType.FLOAT, data.shape)
    for tid in 'efgh':
        ts.addVarInit(tid, tinfo, data)

    actGrads = ts.getIds(TensorType.ActGrad)
    assert len(actGrads) == 4
    assert set(actGrads) == set([i for i in 'abcd'])

    variables = ts.getIds(TensorType.Variable)
    assert len(variables) == 4
    assert set(variables) == set([i for i in 'efgh'])


def test_get_of_type():
    ir = Ir()
    gId = GraphId("g")
    graph = Graph(ir, gId)
    ts = Tensors(graph)

    for tid in 'abcd':
        ts.addActGrad(tid)

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    tinfo = TensorInfo(DataType.FLOAT, data.shape)
    for tid in 'efgh':
        ts.addVarInit(tid, tinfo, data)

    for tid in 'ijkl':
        ts.addStream(tid, tinfo)

    actGrads = ts.getOfType(TensorType.ActGrad)
    assert len(actGrads) == 4
    assert set([i.id for i in actGrads]) == set([i for i in 'abcd'])

    variables = ts.getOfType(TensorType.Variable)
    assert len(variables) == 4
    assert set([i.id for i in variables]) == set([i for i in 'efgh'])

    streams = ts.getOfType(TensorType.Stream)
    assert len(streams) == 4
    assert set([i.id for i in streams]) == set([i for i in 'ijkl'])

    actGradsAndVars = ts.getOfType([TensorType.ActGrad, TensorType.Variable])
    assert len(actGradsAndVars) == 8
    assert set([i.id for i in actGradsAndVars]) == set([i for i in 'abcdefgh'])
