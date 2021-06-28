# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir


def test_graph_construction():
    """ Test that we can construct a popart._internal.ir.Graph object. """
    ir = _ir.Ir()
    gId = _ir.GraphId("g")
    _ = _ir.Graph(ir, gId)
