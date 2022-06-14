# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

import popart
import popxl
import popxl.ops as ops

import popart._internal.ir as _ir
from utils import contains_op_of_type, num_op_of_type
from typing import List


class TestCall:
    def test_call_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        def id_fn(x: popxl.Tensor):
            return x

        with g:
            a = popxl.variable(1)

            id_graph = ir.create_graph(id_fn, a)
            b, = ops.call(id_graph, a)

        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Call", _ir.op.CallOp, g)

        assert len(id_graph.tensors) == 1
        assert len(id_graph.variables) == 0
        # Rudimentarily test subgraph has no ops with negative tests
        assert not contains_op_of_type("Call", _ir.op.CallOp, id_graph)
        assert not contains_op_of_type("Add", _ir.op.AddOp, id_graph)

    def test_call_module_with_internal_inputs_and_multiple_callsites(self):
        ir = popxl.Ir()
        g = ir.main_graph

        class AddWeight(popxl.Module):
            def __init__(self):
                self.w: popxl.Tensor = None

            def build(self, x):
                self.w = popxl.graph_input(x.shape, x.dtype, "w")
                return self.w + x

        with g:
            w0 = popxl.variable(1)
            x0 = popxl.variable(1)

            # First graph
            add_weight0 = AddWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x0)

            with pytest.raises(ValueError):
                # Call without `inputs_dict`
                y0, = ops.call(add_weight_graph0, x0)

            # First call site
            y0, = ops.call(add_weight_graph0,
                           x0,
                           inputs_dict={add_weight0.w: w0})

            # Second call site of same graph
            w1 = popxl.variable(1)
            x1 = popxl.variable(1)

            y1, = ops.call(add_weight_graph0,
                           x1,
                           inputs_dict={add_weight0.w: w1})

            # Second graph from new instance of module.
            # ir.create_graph should be able to create a new unique Graph name.
            add_weight1 = AddWeight()
            add_weight_graph1 = ir.create_graph(add_weight1, x0)

            # Call second graph. Reuse x0 and w1 as inputs.
            y2, = ops.call(add_weight_graph1,
                           x0,
                           inputs_dict={add_weight1.w: w1})

            # Third graph that reuses module add_weight1.
            # This calls `build` again, and thus simply overwrites add_weight1.w
            # to be the tensor in the new subgraph add_weight_graph2.
            old_w1_id = add_weight1.w.id
            add_weight_graph2 = ir.create_graph(add_weight1, x1)

            assert old_w1_id != add_weight1.w.id

            # Call third graph. Reuse x1 and w0 as inputs.
            y3, = ops.call(add_weight_graph2,
                           x1,
                           inputs_dict={add_weight1.w: w0})

        # Test main graph
        # 4 vars + y0 + y1 + y2 + y3
        # 4 call sites total
        assert len(g.tensors) == 8
        assert len(g.variables) == 4
        assert num_op_of_type("Call", _ir.op.CallOp, g) == 5

        # Test subgraphs have unique scopes
        assert add_weight_graph0.name != add_weight_graph1.name
        assert add_weight_graph1.name != add_weight_graph2.name
        assert add_weight_graph0.name != add_weight_graph2.name

        # Test subgraphs (should be identical)

        def test_subgraph(add_weight_subgraph: popxl.Graph):
            assert len(add_weight_subgraph.tensors) == 3
            assert len(add_weight_subgraph.variables) == 0
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

    def test_by_ref(self):
        ir = popxl.Ir()

        def foo(x: popxl.TensorByRef, y: popxl.Tensor):
            return ops.var_updates.accumulate_(x, y)

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)

            g = ir.create_graph(foo, v1, v2)
            info = ops.call_with_info(g, v1, v2)

        assert len(g._by_ref_inputs) == 1
        assert info._op.modifiesIndex(0)
        assert not info._op.modifiesIndex(1)

    def test_can_pass_same_tensor_multiple_times(self):
        ir = popxl.Ir()

        with ir.main_graph:
            a = popxl.variable(1)
            add_g = ir.create_graph(lambda x, y: x + y, a, a)
            b, = ops.call(add_g, a, a)

    def test_mismatch_inputs_error(self):
        ir = popxl.Ir()

        with ir.main_graph:
            a = popxl.variable(1, popxl.float32)
            add_g = ir.create_graph(lambda x: x + 1, a)
            b = popxl.variable(1, popxl.float16)

            with pytest.raises(popart.popart_exception):
                ops.call(add_g, b)

            c = popxl.variable([1, 2], popxl.float32)

            with pytest.raises(popart.popart_exception):
                ops.call(add_g, c)

    def test_input_list_of_tensors(self):
        ir = popxl.Ir()

        def sum_(a: popxl.Tensor, bs: List[popxl.Tensor], c: popxl.Tensor):
            x = a + c
            for b in bs:
                x += b
            return x

        with ir.main_graph:
            a = popxl.variable(1)

            g = ir.create_graph(sum_, a, [a, a, a], a)

            x, = ops.call(g, a, [a, a, a], a)

        assert len(g.inputs) == 5
        assert len(g.outputs) == 1

    def test_coerce_tensor_positionally(self):
        """Auto-convert value "1" into tensor on ops.call using inputs"""
        ir = popxl.Ir()

        def identity_(x):
            return x

        with ir.main_graph:
            a = popxl.variable(1, dtype=popxl.int32)

            g = ir.create_graph(identity_, a.spec)

            x, = ops.call(g, 1)

        assert len(g.inputs) == 1
        assert len(g.outputs) == 1
        assert x.dtype == popxl.int32

    def test_coerce_tensor_inputs_dict(self):
        """Auto-convert value "1" into tensor on ops.call using inputs_dict"""
        ir = popxl.Ir()

        def identity_(x):
            return x

        with ir.main_graph:
            a = popxl.variable(1, dtype=popxl.int32)

            g = ir.create_graph(identity_, a.spec)

            x, = ops.call(g, inputs_dict={g.inputs[0]: 1})

        assert len(g.inputs) == 1
        assert len(g.outputs) == 1
        assert x.dtype == popxl.int32

    def test_error_duplicate_inputs_positional_and_inputs_dict(self):
        """Error if specifing the same input in inputs and inputs_dict"""
        ir = popxl.Ir()

        def identity_(x):
            return x

        with ir.main_graph:
            a = popxl.variable(1, dtype=popxl.int32)

            g = ir.create_graph(identity_, a)

            with pytest.raises(ValueError):
                ops.call(g, a, inputs_dict={g.inputs[0]: a})

    def test_error_could_not_coerce_into_tensor(self):
        """Error if could not auto-convert value into tensor using ops.call"""
        ir = popxl.Ir()

        def identity_(x):
            return x

        with ir.main_graph:
            a = popxl.variable(1, dtype=popxl.int32)

            g = ir.create_graph(identity_, a)

            with pytest.raises(TypeError):
                ops.call(g, 'a')

            with pytest.raises(TypeError):
                ops.call(g, None)
