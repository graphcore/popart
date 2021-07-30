# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Unit tests for the `Ir` class.

This file should contain unittests to check the correct working of the Ir class
from the popart.ir package. This is a public-facing API. Tests in this file
should be quick to run (these are explicitly not integration tests).
"""

import popart.ir as pir


class TestIr:
    def test_constructor(self):
        """Test that the `Ir` constructor sets the internal object state
        correctly.
        """
        ir = pir.Ir()
        # The low-level IR should have only one graph - the main graph.
        assert len(ir._pb_ir.getAllGraphs()) == 1
        # The high-level IR should have no subgraphs.
        assert len(ir._subgraphs) == 0
        # The `_pure_names` counter should be empty.
        assert len(ir._pure_names) == 0

        main_graph = ir.main_graph()
        # The name of the main graph in the high-level IR should be 'main'.
        assert main_graph.name == 'main'
        # The high-level main graph should have a '_pb_graph' attribute that's
        # equal to the low-level main graph instance.
        assert main_graph._pb_graph == ir._pb_ir.getMainGraph()

    def test_get_graph_0(self):
        """Test that the `Ir` class creates `Graph`s using the `get_graph`
        method, that their names are as expected and that they point to unique
        low-level graphs.
        """
        from collections import Counter

        ir = pir.Ir()

        # A python function that will be converted to a graph.
        def foo():
            ...

        for i in range(3):
            foo_graph = ir.get_graph(foo)
            # The number of subgraphs should increase by one.
            assert len(ir._subgraphs) == i + 1
            for j, graph in enumerate(ir._subgraphs):
                # Check if the name of the subgraphs is correctly inferred both
                # in the high and low level.
                name = f'TestIr.test_get_graph_0.foo_{j}'
                assert graph.name == name
                assert graph._pb_graph.id.str() == name
                if i == j:
                    assert foo_graph.name == name

        # Check if the `_pure_names` counter is working as expected.
        assert ir._pure_names == Counter({'TestIr.test_get_graph_0.foo': 3})

        # Check if all graphs in the IR point to unique low-level graphs.
        all_graphs = [ir._main_graph._pb_graph
                      ] + [g._pb_graph for g in ir._subgraphs]
        assert len(all_graphs) == len(set(all_graphs))
