# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Unit tests for the `Ir` class.

This file should contain unittests to check the correct working of the Ir class
from the popart.ir package. This is a public-facing API. Tests in this file
should be quick to run (these are explicitly not integration tests).
"""
import gc
import pytest
import popart.ir as pir
from popart.ir import ops
from typing import List


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


def test_cache():
    # Force the Ir._ir_cache to start empty
    gc.collect()

    ir = pir.Ir()
    ir1 = pir.Ir._from_pb(ir._pb_ir)
    assert ir is ir1

    assert len(pir.Ir._ir_cache) == 1
    del ir
    gc.collect()
    assert len(pir.Ir._ir_cache) == 1
    del ir1
    gc.collect()
    assert len(pir.Ir._ir_cache) == 0


class TestCreateGraph:
    def test_create_graph(self):
        ir = pir.Ir()

        def foo(x: pir.TensorByRef, y: pir.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph():
            v1 = pir.variable(1)
            v2 = pir.variable(2)

            g = ir.create_graph(foo, v1, v2, 5)

            z, = ops.call(g, v1, v2)

        assert len(g._by_ref_inputs) == 1
        x = g.get_input_tensors()[0]
        assert x == g._by_ref_inputs.pop()
        assert x.name == "x"

    def test_create_graph_from_module(self):
        ir = pir.Ir()

        class Bar(pir.Module):
            def __init__(self):
                self.y: pir.Tensor = None

            def build(self, a, b):
                self.y = pir.subgraph_input((1, ), pir.int32, "y")
                return a + b + self.y

        with ir.main_graph():
            a = pir.variable([1])
            b = pir.variable([2])
            y = pir.variable([3])

            bar = Bar()
            g = ir.create_graph(bar, a, b)

            y1, = ops.call(g, a, b, subgraph_in_to_parent_in={bar.y: y})

        assert len(g.get_input_tensors()) == 3
        assert len(g.get_output_tensors()) == 1

    def test_bad_num_arguments(self):
        ir = pir.Ir()

        def foo(x: pir.TensorByRef, y: pir.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph():
            v1 = pir.variable(1)
            v2 = pir.variable(2)
            v3 = pir.variable(2)

            # Missing arguments
            with pytest.raises(ValueError):
                g = ir.create_graph(foo, v1, 5)

            # Too many arguments
            with pytest.raises(ValueError):
                g = ir.create_graph(foo, v1, v2, v3, 5)

    def test_bad_mixed_arguments(self):
        ir = pir.Ir()

        def foo(x):
            pass

        with ir.main_graph():
            v1 = pir.variable(1)

            # Mixed args: Tensor + non-Tensor
            with pytest.raises(TypeError):
                g = ir.create_graph(foo, [v1, 5])

            # Mixed args: non-Tensor + Tensor
            with pytest.raises(TypeError):
                g = ir.create_graph(foo, [5, v1])

    def test_bad_mixed_var_arguments(self):
        ir = pir.Ir()

        def foo(*x):
            pass

        with ir.main_graph():
            v1 = pir.variable(1)

            # Mixed args: Tensor + non-Tensor
            with pytest.raises(TypeError):
                g = ir.create_graph(foo, v1, 5)

            # Mixed args: non-Tensor + Tensor
            with pytest.raises(TypeError):
                g = ir.create_graph(foo, 5, v1)

    def test_default_args(self):
        ir = pir.Ir()

        with ir.main_graph():
            x = pir.variable(1)

            def sum_xab(x, a=3):
                return x + a

            g = ir.create_graph(sum_xab, x)

            y, = ops.call(g, x)

        assert len(g.get_input_tensors()) == 1
        assert len(g.get_output_tensors()) == 1
        assert len(g.get_constants()) == 1

    @pytest.mark.parametrize("type_", [pir.Tensor, pir.TensorByRef])
    def test_variable_args_and_outputs(self, type_):
        ir = pir.Ir()

        def cumulative_sum(*xs: type_):
            outputs = [xs[0]]
            x = xs[0]
            if len(xs) > 1:
                for x_i in xs[1:]:
                    x = x + x_i
                    outputs += [x]
            return outputs

        with ir.main_graph():
            x1 = pir.variable(1)
            x2 = pir.variable(2)

            g = ir.create_graph(cumulative_sum, x1, x2)

            y1, y2 = ops.call(g, x1, x2)

        assert len(g.get_input_tensors()) == 2
        assert len(g.get_output_tensors()) == 2
        if type_ is pir.TensorByRef:
            assert len(g._by_ref_inputs) == 2

    @pytest.mark.parametrize("type_", [pir.Tensor, pir.TensorByRef])
    def test_with_list(self, type_):
        ir = pir.Ir()

        def cumulative_sum(xs: List[type_]):
            outputs = [xs[0]]
            x = xs[0]
            if len(xs) > 1:
                for x_i in xs[1:]:
                    x = x + x_i
                    outputs += [x]
            return outputs

        with ir.main_graph():
            x1 = pir.variable(1)
            x2 = pir.variable(2)

            g = ir.create_graph(cumulative_sum, [x1, x2])

            y1, y2 = ops.call(g, [x1, x2])

        assert len(g.get_input_tensors()) == 2
        assert len(g.get_output_tensors()) == 2
        if type_ is pir.TensorByRef:
            assert len(g._by_ref_inputs) == 2

    @pytest.mark.parametrize("type_", [pir.Tensor, pir.TensorByRef])
    def test_with_variable_kwargs_tensors(self, type_):
        ir = pir.Ir()

        def cumulative_sum(**xs: type_):
            xs = list(xs.values())
            outputs = [xs[0]]
            x = xs[0]
            if len(xs) > 1:
                for x_i in xs[1:]:
                    x = x + x_i
                    outputs += [x]
            return outputs

        with ir.main_graph():
            x1 = pir.variable(1)
            x2 = pir.variable(2)

            g = ir.create_graph(cumulative_sum, x1=x1, x2=x2)

            y1, y2 = ops.call(g, x1, x2)

        assert len(g.get_input_tensors()) == 2
        assert len(g.get_output_tensors()) == 2
        if type_ is pir.TensorByRef:
            assert len(g._by_ref_inputs) == 2

    def test_complicated_signature(self):
        ir = pir.Ir()

        def sum_all(a,
                    b: List[pir.Tensor],
                    c: bool,
                    *args: pir.TensorByRef,
                    e: pir.Tensor,
                    f: int = 0,
                    **kwargs: pir.Tensor):
            x = a + f
            for t in b + list(args) + [e] + list(kwargs.values()):
                x += t
            return x

        with ir.main_graph():
            x = [pir.variable(1) for i in range(8)]

            g = ir.create_graph(sum_all,
                                x[0], [x[1], x[2]],
                                True,
                                x[3],
                                x[4],
                                e=x[5],
                                x=x[6],
                                z=x[7])

            y, = ops.call(g, *x)

        assert len(g.get_input_tensors()) == len(x)
        assert len(g.get_output_tensors()) == 1
        assert len(g._by_ref_inputs) == 2

    def test_bad_output(self):
        ir = pir.Ir()

        def fun():
            return True

        with ir.main_graph():
            with pytest.raises(ValueError):
                g = ir.create_graph(fun)

    def test_bad_list_output(self):
        ir = pir.Ir()

        def fun():
            return [True, True]

        with ir.main_graph():
            with pytest.raises(ValueError):
                g = ir.create_graph(fun)

    def test_create_graph_tensor_spec(self):
        ir = pir.Ir()

        def foo(x: pir.TensorByRef, y: pir.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph():
            v1 = pir.variable(1)
            v2 = pir.variable(2)

            g = ir.create_graph(foo, v1.tensor_spec, v2.tensor_spec, 5)

        assert len(g._by_ref_inputs) == 1
        x = g.get_input_tensors()[0]
        assert x == g._by_ref_inputs.pop()
        assert x.name == "x"

    def test_create_graph_tensor_spec_standalone(self):
        ir = pir.Ir()

        def foo(x: pir.TensorByRef, y: pir.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph():
            g = ir.create_graph(foo, pir.TensorSpec((), pir.int32),
                                pir.TensorSpec((), pir.int32), 5)

        assert len(g._by_ref_inputs) == 1
        x = g.get_input_tensors()[0]
        assert x == g._by_ref_inputs.pop()
        assert x.name == "x"
