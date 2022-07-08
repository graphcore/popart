# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Unit tests for the `Ir` class.

This file should contain unittests to check the correct working of the Ir class
from the popxl package. This is a public-facing API. Tests in this file
should be quick to run (these are explicitly not integration tests).
"""
import gc
import pytest
import popxl
from popxl import ops
from typing import List


def test_constructor():
    """Test that the `Ir` constructor sets the internal object state
    correctly.
    """
    ir = popxl.Ir()
    # The low-level IR should have only one graph - the main graph.
    assert len(ir._pb_ir.getAllGraphs()) == 1

    main_graph = ir.main_graph
    assert isinstance(main_graph, popxl.Graph)


def test_multiple_ir():
    ir1 = popxl.Ir()
    ir2 = popxl.Ir()
    assert ir1.id != ir2.id
    assert ir1 != ir2  # test __eq__
    assert len(set([ir1, ir2])) == 2  # test __hash__


def test_repr():
    ir1 = popxl.Ir()
    str(ir1)


def test_cache():
    # Force the Ir._ir_cache to start empty
    gc.collect()

    ir = popxl.Ir()
    ir1 = popxl.Ir._from_pb(ir._pb_ir)
    assert ir is ir1

    assert len(popxl.Ir._ir_cache) == 1
    del ir
    gc.collect()
    assert len(popxl.Ir._ir_cache) == 1
    del ir1
    gc.collect()
    assert len(popxl.Ir._ir_cache) == 0


class TestCreateGraph:
    def test_create_graph(self):
        ir = popxl.Ir()

        def foo(x: popxl.TensorByRef, y: popxl.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)

            g = ir.create_graph(foo, v1, v2, 5)

            (_,) = ops.call(g, v1, v2)

        assert len(g._by_ref_inputs) == 1
        x = g.inputs[0]
        assert x == g._by_ref_inputs.pop()
        assert x.name == "x"

    def test_create_graph_from_module(self):
        ir = popxl.Ir()

        class Bar(popxl.Module):
            def __init__(self):
                self.y: popxl.Tensor = None

            def build(self, a, b):
                self.y = popxl.graph_input((1,), popxl.int32, "y")
                return a + b + self.y

        with ir.main_graph:
            a = popxl.variable([1])
            b = popxl.variable([2])
            y = popxl.variable([3])

            bar = Bar()
            g = ir.create_graph(bar, a, b)

            (_,) = ops.call(g, a, b, inputs_dict={bar.y: y})

        assert len(g.inputs) == 3
        assert len(g.outputs) == 1

    def test_bad_num_arguments(self):
        ir = popxl.Ir()

        def foo(x: popxl.TensorByRef, y: popxl.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)
            v3 = popxl.variable(2)

            # Missing arguments
            with pytest.raises(ValueError):
                _ = ir.create_graph(foo, v1, 5)

            # Too many arguments
            with pytest.raises(ValueError):
                _ = ir.create_graph(foo, v1, v2, v3, 5)

    def test_bad_mixed_arguments(self):
        ir = popxl.Ir()

        def foo(_):  # x is an unused argument
            pass

        with ir.main_graph:
            v1 = popxl.variable(1)

            # Mixed args: Tensor + non-Tensor
            with pytest.raises(TypeError):
                _ = ir.create_graph(foo, [v1, 5])

            # Mixed args: non-Tensor + Tensor
            with pytest.raises(TypeError):
                _ = ir.create_graph(foo, [5, v1])

    def test_bad_mixed_var_arguments(self):
        ir = popxl.Ir()

        def foo(*_):  # x is an unused argument
            pass

        with ir.main_graph:
            v1 = popxl.variable(1)

            # Mixed args: Tensor + non-Tensor
            with pytest.raises(TypeError):
                _ = ir.create_graph(foo, v1, 5)

            # Mixed args: non-Tensor + Tensor
            with pytest.raises(TypeError):
                _ = ir.create_graph(foo, 5, v1)

    def test_default_args(self):
        ir = popxl.Ir()

        with ir.main_graph:
            x = popxl.variable(1)

            def sum_xab(x, a=3):
                return x + a

            g = ir.create_graph(sum_xab, x)

            (_,) = ops.call(g, x)

        assert len(g.inputs) == 1
        assert len(g.outputs) == 1
        assert len(g.constants) == 1

    @pytest.mark.parametrize("type_", [popxl.Tensor, popxl.TensorByRef])
    def test_variable_args_and_outputs(self, type_):
        ir = popxl.Ir()

        def cumulative_sum(*xs: type_):
            outputs = [xs[0]]
            x = xs[0]
            if len(xs) > 1:
                for x_i in xs[1:]:
                    x = x + x_i
                    outputs += [x]
            return outputs

        with ir.main_graph:
            x1 = popxl.variable(1)
            x2 = popxl.variable(2)

            g = ir.create_graph(cumulative_sum, x1, x2)

            _, _ = ops.call(g, x1, x2)

        assert len(g.inputs) == 2
        assert len(g.outputs) == 2
        if type_ is popxl.TensorByRef:
            assert len(g._by_ref_inputs) == 2

    @pytest.mark.parametrize("type_", [popxl.Tensor, popxl.TensorByRef])
    def test_with_list(self, type_):
        ir = popxl.Ir()

        def cumulative_sum(xs: List[type_]):
            outputs = [xs[0]]
            x = xs[0]
            if len(xs) > 1:
                for x_i in xs[1:]:
                    x = x + x_i
                    outputs += [x]
            return outputs

        with ir.main_graph:
            x1 = popxl.variable(1)
            x2 = popxl.variable(2)

            g = ir.create_graph(cumulative_sum, [x1, x2])

            _, _ = ops.call(g, [x1, x2])

        assert len(g.inputs) == 2
        assert len(g.outputs) == 2
        if type_ is popxl.TensorByRef:
            assert len(g._by_ref_inputs) == 2

    @pytest.mark.parametrize("type_", [popxl.Tensor, popxl.TensorByRef])
    def test_with_variable_kwargs_tensors(self, type_):
        ir = popxl.Ir()

        def cumulative_sum(**xs: type_):
            xs = list(xs.values())
            outputs = [xs[0]]
            x = xs[0]
            if len(xs) > 1:
                for x_i in xs[1:]:
                    x = x + x_i
                    outputs += [x]
            return outputs

        with ir.main_graph:
            x1 = popxl.variable(1)
            x2 = popxl.variable(2)

            g = ir.create_graph(cumulative_sum, x1=x1, x2=x2)

            _, _ = ops.call(g, x1, x2)

        assert len(g.inputs) == 2
        assert len(g.outputs) == 2
        if type_ is popxl.TensorByRef:
            assert len(g._by_ref_inputs) == 2

    def test_complicated_signature(self):
        ir = popxl.Ir()

        def sum_all(
            a,
            b: List[popxl.Tensor],
            _: bool,  # c is an unused argument
            *args: popxl.TensorByRef,
            e: popxl.Tensor,
            f: int = 0,
            **kwargs: popxl.Tensor
        ):
            x = a + f
            for t in b + list(args) + [e] + list(kwargs.values()):
                x += t
            return x

        with ir.main_graph:
            x = [popxl.variable(1) for i in range(8)]

            g = ir.create_graph(
                sum_all, x[0], [x[1], x[2]], True, x[3], x[4], e=x[5], x=x[6], z=x[7]
            )

            (_,) = ops.call(g, *x)

        assert len(g.inputs) == len(x)
        assert len(g.outputs) == 1
        assert len(g._by_ref_inputs) == 2

    def test_bad_output(self):
        ir = popxl.Ir()

        def fun():
            return True

        with ir.main_graph:
            with pytest.raises(ValueError):
                _ = ir.create_graph(fun)

    def test_bad_list_output(self):
        ir = popxl.Ir()

        def fun():
            return [True, True]

        with ir.main_graph:
            with pytest.raises(ValueError):
                _ = ir.create_graph(fun)

    def test_create_graph_tensor_spec(self):
        ir = popxl.Ir()

        def foo(x: popxl.TensorByRef, y: popxl.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)

            g = ir.create_graph(foo, v1.spec, v2.spec, 5)

        assert len(g._by_ref_inputs) == 1
        x = g.inputs[0]
        assert x == g._by_ref_inputs.pop()
        assert x.name == "x"

    def test_create_graph_tensor_spec_standalone(self):
        ir = popxl.Ir()

        def foo(x: popxl.TensorByRef, y: popxl.Tensor, c: int):
            return (x * c) + y

        with ir.main_graph:
            g = ir.create_graph(
                foo,
                popxl.TensorSpec((), popxl.int32),
                popxl.TensorSpec((), popxl.int32),
                5,
            )

        assert len(g._by_ref_inputs) == 1
        x = g.inputs[0]
        assert x == g._by_ref_inputs.pop()
        assert x.name == "x"

    def test_create_graph_tensor_spec_list(self):
        ir = popxl.Ir()

        def foo(x: popxl.TensorByRef, ys: List[popxl.Tensor], c: int):
            return [(x * c) + y for y in ys]

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)

            g = ir.create_graph(foo, v1.spec, [v2.spec, v2.spec], 5)

        assert len(g.inputs) == 3
        assert len(g.outputs) == 2
        assert len(g._by_ref_inputs) == 1


def test_num_host_transfers_property():
    ir = popxl.Ir()
    di = ir.num_host_transfers
    dj = di + 10
    ir.num_host_transfers = dj
    assert ir.num_host_transfers != di
    assert ir.num_host_transfers == dj


def test_get_all_d2h_streams():

    ir = popxl.Ir()

    with ir.main_graph:
        v = popxl.variable(1)
        d = popxl.d2h_stream(v.shape, v.dtype)
        ops.host_store(d, v)
        e = popxl.d2h_stream(v.shape, v.dtype)
        ops.host_store(e, v)
        f = popxl.d2h_stream(v.shape, v.dtype)
        h = popxl.h2d_stream(v.shape, v.dtype)
        w = v + 1

    expected_d2h_ids = [d.tensor_id, e.tensor_id]
    # No ops.host_store along stream f, therefore it is not expected.
    unexpected_d2h_ids = [f.tensor_id, w.id, v.id, h.tensor_id]

    d2hs = ir.get_all_d2h_streams()

    assert len(d2hs) == 2
    _pb_mg = ir._pb_ir.getMainGraph()
    for d2h in d2hs:
        assert d2h.tensor_id in expected_d2h_ids
        assert d2h.tensor_id not in unexpected_d2h_ids
        assert d2h.tensor_id in _pb_mg


def test_get_all_h2d_streams():

    ir = popxl.Ir()

    with ir.main_graph:
        v = popxl.variable(1)
        d = popxl.h2d_stream(v.shape, v.dtype)
        a = ops.host_load(d)
        e = popxl.h2d_stream(v.shape, v.dtype)
        b = ops.host_load(e)
        f = popxl.h2d_stream(v.shape, v.dtype)
        g = popxl.d2h_stream(v.shape, v.dtype)
        w = v + a + b
        ops.host_store(g, w)

    expected_h2d_ids = [d.tensor_id, e.tensor_id]
    # No ops.host_store along stream f, therefore it is not expected.
    unexpected_h2d_ids = [w.id, v.id, a.id, b.id, f.tensor_id, g.tensor_id]

    h2ds = ir.get_all_h2d_streams()

    assert len(h2ds) == 2
    _pb_mg = ir._pb_ir.getMainGraph()
    for h2d in h2ds:
        assert h2d.tensor_id not in unexpected_h2d_ids
        assert h2d.tensor_id in expected_h2d_ids
        assert h2d.tensor_id in _pb_mg
