# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import gc
import os
import popart
import popart.ir as pir


def test_basic(tmpdir):
    # debugInfo is written in the destructor.
    # Ensure there are no uncollected objects before initialising
    gc.collect()
    filename = os.path.join(tmpdir, "debug.json")
    popart.initializePoplarDebugInfo(filename, "json")

    ir = pir.Ir()
    with ir.main_graph():
        a = pir.variable(0)
        b = pir.variable(1)
        c = a + b

    gc.collect()
    popart.closePoplarDebugInfo()

    with open(filename) as json_file:
        data = json.load(json_file)

    ctxs = data["contexts"]
    assert len(ctxs) == 4  # 1 op, 3 tensors
    op_ctx = None
    for ctx in ctxs:
        if ctx['category'] == 'op':
            op_ctx = ctx
            break
    assert op_ctx is not None
    assert op_ctx["layer"] == 'popart.ir'
    assert op_ctx["api"] == 'add'
    assert set(op_ctx["inputs"]) == {a.id, b.id}
    assert set(op_ctx["outputs"]) == {c.id}
    assert op_ctx["location"]["lineNumber"] == 20
    assert __file__ == data["stringTable"][op_ctx["location"]["fileName"]]
    assert 'test_basic' == data["stringTable"][op_ctx["location"]
                                               ["functionName"]]


def test_namescoping():
    ir = pir.Ir()
    with ir.main_graph(), pir.name_scope("foo"):
        a = pir.variable(1, name="bar")
    assert a.name == "foo/bar"
