# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import gc
import os
import pathlib
import popart
import popxl


def test_basic(tmpdir):
    # debugInfo is written in the destructor.
    # Ensure there are no uncollected objects before initialising
    gc.collect()
    filename = os.path.join(tmpdir, "debug.json")
    popart.initializePoplarDebugInfo(filename, "json")

    ir = popxl.Ir()
    with ir.main_graph:
        a = popxl.variable(0)
        b = popxl.variable(1)
        c = a + b

    gc.collect()
    popart.closePoplarDebugInfo()

    with open(filename, encoding="utf-8") as json_file:
        data = json.load(json_file)

    ctxs = data["contexts"]
    assert len(ctxs) == 4  # 1 op, 3 tensors
    op_ctx = None
    for ctx in ctxs:
        if ctx['category'] == 'op':
            op_ctx = ctx
            break
    assert op_ctx is not None
    assert op_ctx["layer"] == 'popxl'
    assert op_ctx["api"] == 'add'
    assert set(op_ctx["inputs"]) == {a.id, b.id}
    assert set(op_ctx["outputs"]) == {c.id}
    assert op_ctx["location"]["lineNumber"] == 21
    # Use pathlib to resolve any symlinks
    assert str(pathlib.Path(__file__).resolve()) == data["stringTable"][
        op_ctx["location"]["fileName"]]
    assert 'test_basic' == data["stringTable"][op_ctx["location"]
                                               ["functionName"]]


def test_namescoping():
    ir = popxl.Ir()
    with ir.main_graph, popxl.name_scope("foo"):
        a = popxl.variable(1, name="bar")
    assert a.name == "foo/bar"
