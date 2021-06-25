# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir


def test_graphid_construction():
    """ Test that we can construct a popart._internal.ir.GraphId object. """
    _ir.GraphId("g")
