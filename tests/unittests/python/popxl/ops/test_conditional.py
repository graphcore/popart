# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import List
import pytest

import popart._internal.ir as _ir

import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestConditional:
    def test_fn(self):
        ir = popxl.Ir()
        graph = ir.main_graph

        def then_fn(x: popxl.Tensor):
            return x

        def else_fn(y: popxl.Tensor):
            return y + 1

        with graph:
            cond = popxl.variable(1, popxl.bool, name="cond")
            a = popxl.variable(1, popxl.float32, name="a")
            b = popxl.variable(2, popxl.float32, name="b")
            graph_then = ir.create_graph(then_fn, a)
            graph_else = ir.create_graph(else_fn, b)
            b = ops.conditional(cond, graph_then, graph_else, [a], [b])
        assert len(graph.tensors) == 4
        assert len(graph.variables) == 3
        assert contains_op_of_type("If", _ir.op.IfOp, graph)

    def test_without_inputs_dict(self):
        ir = popxl.Ir()
        graph = ir.main_graph

        class AddWeight(popxl.Module):
            def __init__(self):
                self.w: popxl.Tensor = None

            def build(self, x):
                self.w = popxl.graph_input(x.shape, x.dtype, "w")
                return self.w + x

        with graph:
            x0 = popxl.variable(1, name="x0")
            cond = popxl.variable(True, name="cond")

            add_weight0 = AddWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x0)

            with pytest.raises(ValueError):
                # without `inputs_dict`
                y0, = ops.conditional(cond=cond,
                                      then_branch=add_weight_graph0,
                                      else_branch=add_weight_graph0,
                                      then_inputs=[x0],
                                      else_inputs=[x0])

    def test_error_duplicate_inputs_positional_and_inputs_dict(self):
        """Error if specifing the same input in inputs and inputs_dict"""
        ir = popxl.Ir()

        def identity_(x):
            return x

        with ir.main_graph:
            a = popxl.variable(1, dtype=popxl.int32)
            g = ir.create_graph(identity_, a)
            cond = popxl.variable(True, name="cond")

            with pytest.raises(ValueError):
                y0, = ops.conditional(cond=cond,
                                      then_branch=g,
                                      else_branch=g,
                                      then_inputs=[a],
                                      else_inputs=[a],
                                      then_inputs_dict={g.inputs[0]: a},
                                      else_inputs_dict={g.inputs[0]: a})

    def test_error_could_not_coerce_into_tensor(self):
        """Error if could not auto-convert value into tensor using ops.conditional"""
        ir = popxl.Ir()

        def identity_(x):
            return x

        with ir.main_graph:
            a = popxl.variable(1, dtype=popxl.int32)
            g = ir.create_graph(identity_, a)
            cond = popxl.variable(True, name="cond")

            with pytest.raises(TypeError):
                ops.conditional(cond=cond,
                                then_branch=g,
                                else_branch=g,
                                then_inputs=['a'],
                                else_inputs=['a'])

            with pytest.raises(ValueError):
                ops.conditional(cond=cond,
                                then_branch=g,
                                else_branch=g,
                                then_inputs=None,
                                else_inputs=None)

    # TODO T66328
    # def test_by_ref(self):
    #     ir = popxl.Ir()
    #
    #     # x passed by ref
    #     def foo(x: popxl.TensorByRef, y: popxl.Tensor):
    #         return ops.var_updates.accumulate_(x, y)
    #
    #     # x passed not by ref
    #     def bar(x, y: popxl.Tensor):
    #         return ops.var_updates.accumulate_(x, y)
    #
    #     with ir.main_graph:
    #         v1 = popxl.variable(1)
    #         v2 = popxl.variable(2)
    #         cond = popxl.variable(True, name="cond")
    #
    #         foo_g = ir.create_graph(foo, v1, v2)
    #         bar_g = ir.create_graph(bar, v1, v2)
    #
    #         call_info = ops.conditional_with_info(cond=cond,
    #                         then_branch=foo_g,
    #                         else_branch=bar_g,
    #                         then_inputs=(v1, v2),
    #                         else_inputs=(v1, v2))
    #
    #     assert len(foo_g._by_ref_inputs) == 1
    #     assert len(bar_g._by_ref_inputs) == 0
    #     assert not call_info._op.modifiesIndex(0) # cond
    #     assert call_info._op.modifiesIndex(1) # foo: v1
    #     assert not call_info._op.modifiesIndex(2) # foo: v2
    #     assert not call_info._op.modifiesIndex(3) # bar: v1
    #     assert not call_info._op.modifiesIndex(4) # bar: v2

    def test_by_ref_error(self):
        ir = popxl.Ir()

        # x passed by ref
        def foo(x: popxl.TensorByRef, y: popxl.Tensor):
            return ops.var_updates.accumulate_(x, y)

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)
            cond = popxl.variable(True, name="cond")

            foo_g = ir.create_graph(foo, v1, v2)

            with pytest.raises(NotImplementedError):
                ops.conditional_with_info(cond=cond,
                                          then_branch=foo_g,
                                          else_branch=foo_g,
                                          then_inputs=(v1, v2),
                                          else_inputs=(v1, v2))

    def test_input_list_of_tensors(self):
        ir = popxl.Ir()

        def sum_(a: popxl.Tensor, bs: List[popxl.Tensor], c: popxl.Tensor):
            x = a + c
            for b in bs:
                x += b
            return x

        with ir.main_graph:
            a = popxl.variable(1)
            cond = popxl.variable(True, name="cond")

            g = ir.create_graph(sum_, a, [a, a, a], a)

            ops.conditional_with_info(cond=cond,
                                      then_branch=g,
                                      else_branch=g,
                                      then_inputs=(a, [a, a, a], a),
                                      else_inputs=(a, [a, a, a], a))

        assert len(g.inputs) == 5
        assert len(g.outputs) == 1

    def test_multiple_outputs(self):
        ir = popxl.Ir()

        def foo(a: popxl.Tensor):
            return a + 1, a + 2

        with ir.main_graph:
            a = popxl.variable(1)
            cond = popxl.variable(True, name="cond")

            foo_g = ir.create_graph(foo, a)

            call_info = ops.conditional_with_info(cond=cond,
                                                  then_branch=foo_g,
                                                  else_branch=foo_g,
                                                  then_inputs=[a],
                                                  else_inputs=[a])

        assert len(call_info.outputs) == 2

    def test_zero_outputs_and_zero_inputs(self):
        ir = popxl.Ir()

        def foo():
            return

        with ir.main_graph:
            cond = popxl.variable(True, name="cond")

            foo_g = ir.create_graph(foo)

            call_info = ops.conditional_with_info(cond=cond,
                                                  then_branch=foo_g,
                                                  else_branch=foo_g)

        assert len(call_info.outputs) == 0

    def test_error_different_number_of_outputs(self):
        ir = popxl.Ir()

        def foo(a: popxl.Tensor):
            return a + 1

        def bar(a: popxl.Tensor):
            return a + 1, a + 2

        with ir.main_graph:
            a = popxl.variable(1)
            cond = popxl.variable(True, name="cond")

            foo_g = ir.create_graph(foo, a)
            bar_g = ir.create_graph(bar, a)

            with pytest.raises(ValueError):
                call_info = ops.conditional_with_info(cond=cond,
                                                      then_branch=foo_g,
                                                      else_branch=bar_g,
                                                      then_inputs=[a],
                                                      else_inputs=[a])


class TestConditionalCallSiteInfo:
    @staticmethod
    def foo(a, b):
        return a + b, a - b

    @staticmethod
    def bar(b, a):
        return a * b, a / b

    def test_called_graph(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert len(call_info.called_graph) == 2
            assert call_info.called_graph[0] == foo_g
            assert call_info.called_graph[1] == bar_g

    def test_graph_to_parent_input_index(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.graph_to_parent_input_index(0, True) == 1
            assert call_info.graph_to_parent_input_index(1, True) == 2
            assert call_info.graph_to_parent_input_index(0, False) == 3
            assert call_info.graph_to_parent_input_index(1, False) == 4

            with pytest.raises(IndexError):
                call_info.graph_to_parent_input_index(2, True)
            with pytest.raises(IndexError):
                call_info.graph_to_parent_input_index(2, False)

    def test_parent_to_graph_input_index(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.parent_to_graph_input_index(1) == 0
            assert call_info.parent_to_graph_input_index(2) == 1
            assert call_info.parent_to_graph_input_index(3) == 0
            assert call_info.parent_to_graph_input_index(4) == 1

            with pytest.raises(IndexError):
                call_info.parent_to_graph_input_index(5)

    def test_is_parent_index_in_then_branch(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.is_parent_index_in_then_branch(1) == True
            assert call_info.is_parent_index_in_then_branch(2) == True
            assert call_info.is_parent_index_in_then_branch(3) == False
            assert call_info.is_parent_index_in_then_branch(4) == False

            with pytest.raises(IndexError):
                call_info.is_parent_index_in_then_branch(5)

    def test_graph_to_parent_output_index(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.graph_to_parent_output_index(0, True) == 0
            assert call_info.graph_to_parent_output_index(1, True) == 1
            assert call_info.graph_to_parent_output_index(0, False) == 0
            assert call_info.graph_to_parent_output_index(1, False) == 1

            with pytest.raises(IndexError):
                call_info.graph_to_parent_output_index(3, True)
            with pytest.raises(IndexError):
                call_info.graph_to_parent_output_index(3, False)

    def test_parent_to_graph_output_index(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.parent_to_graph_output_index(0, True) == 0
            assert call_info.parent_to_graph_output_index(1, True) == 1
            assert call_info.parent_to_graph_output_index(0, False) == 0
            assert call_info.parent_to_graph_output_index(1, False) == 1

            with pytest.raises(IndexError):
                call_info.parent_to_graph_output_index(3, True)
            with pytest.raises(IndexError):
                call_info.parent_to_graph_output_index(3, False)

    def test_graph_to_parent(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.graph_to_parent(
                foo_g.inputs[0]) == call_info.inputs[1]
            assert call_info.graph_to_parent(
                foo_g.inputs[1]) == call_info.inputs[2]
            assert call_info.graph_to_parent(
                bar_g.inputs[0]) == call_info.inputs[3]
            assert call_info.graph_to_parent(
                bar_g.inputs[1]) == call_info.inputs[4]
            assert call_info.graph_to_parent(
                foo_g.outputs[0]) == call_info.outputs[0]
            assert call_info.graph_to_parent(
                foo_g.outputs[1]) == call_info.outputs[1]
            assert call_info.graph_to_parent(
                bar_g.outputs[0]) == call_info.outputs[0]
            assert call_info.graph_to_parent(
                bar_g.outputs[1]) == call_info.outputs[1]

            other_tensor = popxl.variable(1, name='other_tensor')
            with pytest.raises(ValueError):
                call_info.graph_to_parent(other_tensor)

    def test_parent_to_graph(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            c = popxl.variable(1, name='c')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, c, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(c, a))
            assert call_info.parent_to_graph(
                call_info.inputs[1]) == foo_g.inputs[0]
            assert call_info.parent_to_graph(
                call_info.inputs[2]) == foo_g.inputs[1]
            assert call_info.parent_to_graph(
                call_info.inputs[3]) == bar_g.inputs[0]
            # Here it returns the first time `a` was used as an input
            assert call_info.parent_to_graph(
                call_info.inputs[4]) == foo_g.inputs[0]

            other_tensor = popxl.variable(1, name='other_tensor')
            with pytest.raises(ValueError):
                call_info.parent_to_graph(other_tensor)

    def test_parent_input(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.parent_input(0) == cond
            assert call_info.parent_input(1) == a
            assert call_info.parent_input(2) == b
            assert call_info.parent_input(3) == b
            assert call_info.parent_input(4) == a

            with pytest.raises(IndexError):
                call_info.parent_input(5)

    def test_parent_output(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert call_info.parent_output(0) == call_info.outputs[0]
            assert call_info.parent_output(1) == call_info.outputs[1]

            with pytest.raises(IndexError):
                call_info.parent_output(2)

    def test_inputs(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert len(call_info.inputs) == 5
            assert call_info.inputs[0] == cond
            assert call_info.inputs[1] == a
            assert call_info.inputs[2] == b
            assert call_info.inputs[3] == b
            assert call_info.inputs[4] == a

    def test_outputs(self):
        ir = popxl.Ir()
        with ir.main_graph:
            cond = popxl.variable(True, name="cond")
            a = popxl.variable(1, name='a')
            b = popxl.variable(1, name='b')
            foo_g = ir.create_graph(self.foo, a, b)
            bar_g = ir.create_graph(self.bar, b, a)

            call_info = ops.conditional_with_info(cond,
                                                  foo_g,
                                                  bar_g,
                                                  then_inputs=(a, b),
                                                  else_inputs=(b, a))
            assert len(call_info.outputs) == 2
