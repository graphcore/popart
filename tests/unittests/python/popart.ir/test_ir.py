# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Unit tests for the `Ir` class.

This file should contain unittests to check the correct working of the Ir class
from the popart.ir package. This is a public-facing API. Tests in this file
should be quick to run (these are explicitly not integration tests).
"""

import popart.ir as pir


def test_constructor():
    """Test that the `Ir` constructor sets the internal object state
    correctly.
    """
    ir = pir.Ir()
    # The low-level IR should have only one graph - the main graph.
    assert len(ir._pb_ir.getAllGraphs()) == 1

    main_graph = ir.main_graph()
    assert isinstance(main_graph, pir.Graph)


def test_multiple_ir():
    ir1 = pir.Ir()
    ir2 = pir.Ir()
    assert ir1.id != ir2.id
    assert ir1 != ir2  # test __eq__
    assert len(set([ir1, ir2])) == 2  # test __hash__


def test_repr():
    ir1 = pir.Ir()
    str(ir1)
