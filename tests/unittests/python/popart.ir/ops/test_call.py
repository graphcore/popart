# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type, num_op_of_type


class TestCall:
    def test_call_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        def id_fn(x: pir.Tensor):
            return x

        with g:
            a = pir.variable(1)

            id_graph = ir.create_graph(id_fn, a)
            b = ops.call(id_graph, a)

        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Call", _ir.op.CallOp, g)

        assert len(id_graph.get_tensors()) == 1
        assert len(id_graph.get_variables()) == 0
        # Rudimentarily test subgraph has no ops with negative tests
        assert not contains_op_of_type("Call", _ir.op.CallOp, id_graph)
        assert not contains_op_of_type("Add", _ir.op.AddOp, id_graph)

    def test_call_module_with_internal_inputs_and_multiple_callsites(self):
        ir = pir.Ir()
        g = ir.main_graph()

        class AddWeight(pir.Module):
            def __init__(self):
                self.w: pir.Tensor = None

            def build(self, x):
                self.w = pir.subgraph_input(x.shape, x.dtype, "w")
                return self.w + x

        with g:
            w0 = pir.variable(1)
            x0 = pir.variable(1)

            # First graph
            add_weight0 = AddWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x0)

            # First call site
            y0 = ops.call(add_weight_graph0,
                          x0,
                          subgraph_in_to_parent_in={add_weight0.w: w0})

            # Second call site of same graph
            w1 = pir.variable(1)
            x1 = pir.variable(1)

            y1 = ops.call(add_weight_graph0,
                          x1,
                          subgraph_in_to_parent_in={add_weight0.w: w1})

            # Second graph from new instance of module.
            # ir.create_graph should be able to create a new unique Graph name.
            add_weight1 = AddWeight()
            add_weight_graph1 = ir.create_graph(add_weight1, x0)

            # Call second graph. Reuse x0 and w1 as inputs.
            y2 = ops.call(add_weight_graph1,
                          x0,
                          subgraph_in_to_parent_in={add_weight1.w: w1})

            # Third graph that reuses module add_weight1.
            # This calls `build` again, and thus simply overwrites add_weight1.w
            # to be the tensor in the new subgraph add_weight_graph2.
            old_w1_id = add_weight1.w.id
            add_weight_graph2 = ir.create_graph(add_weight1, x1)

            assert old_w1_id != add_weight1.w.id

            # Call third graph. Reuse x1 and w0 as inputs.
            y3 = ops.call(add_weight_graph2,
                          x1,
                          subgraph_in_to_parent_in={add_weight1.w: w0})

        # Test main graph
        # 4 vars + y0 + y1 + y2 + y4
        # 4 call sites total
        assert len(g.get_tensors()) == 8
        assert len(g.get_variables()) == 4
        assert num_op_of_type("Call", _ir.op.CallOp, g) == 4

        # Test subgraphs have unique scopes
        assert add_weight_graph0.name != add_weight_graph1.name
        assert add_weight_graph1.name != add_weight_graph2.name
        assert add_weight_graph0.name != add_weight_graph2.name

        # Test subgraphs (should be identical)

        def test_subgraph(add_weight_subgraph: pir.Graph):
            assert len(add_weight_subgraph.get_tensors()) == 3
            assert len(add_weight_subgraph.get_variables()) == 0
            assert contains_op_of_type("Add", _ir.op.AddOp,
                                       add_weight_subgraph)
            # Rudimentarily test subgraph has only expected ops with negative tests
            assert not contains_op_of_type("Call", _ir.op.CallOp,
                                           add_weight_subgraph)
            assert not contains_op_of_type("Mul", _ir.op.MulOp,
                                           add_weight_subgraph)

        test_subgraph(add_weight_graph0)
        test_subgraph(add_weight_graph1)
        test_subgraph(add_weight_graph2)


# TODO: Test nested subgraphs
