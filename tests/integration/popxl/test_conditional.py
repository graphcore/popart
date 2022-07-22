# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
from popxl.tensor import Tensor
import pytest


class AddSingleConstant(popxl.Module):
    def __init__(self):
        pass

    def build(self, x: Tensor) -> Tensor:
        return x + 5.0


class AddMultipleConstant(popxl.Module):
    def __init__(self):
        pass

    def build(self, x: Tensor) -> Tensor:
        return x + 5.0, x + 10.0


class SubtractSingleConstant(popxl.Module):
    def __init__(self):
        pass

    def build(self, y: Tensor) -> Tensor:
        return y - 3.0


class SubtractMultipleConstant(popxl.Module):
    def __init__(self):
        pass

    def build(self, y: Tensor) -> Tensor:
        return y - 3.0, y - 6.0


class MultiplySingleVariable(popxl.Module):
    def __init__(self):
        pass

    def build(self, x: Tensor, y: Tensor) -> Tensor:
        return x * y


class MultiplyMultipleVariable(popxl.Module):
    def __init__(self):
        pass

    def build(self, x: Tensor, y: Tensor) -> Tensor:
        return x * y, 2 * x * y


class DivideSingleVariable(popxl.Module):
    def __init__(self):
        pass

    def build(self, x: Tensor, y: Tensor) -> Tensor:
        return x / y


class DivideMultipleVariable(popxl.Module):
    def __init__(self):
        pass

    def build(self, x: Tensor, y: Tensor) -> Tensor:
        return x / y, 2 * x / y


class AddSingleWeight(popxl.Module):
    def __init__(self):
        self.w: Tensor = None

    def build(self, x):
        self.w = popxl.graph_input(x.shape, x.dtype, "w")
        return self.w + x


class AddMultipleWeight(popxl.Module):
    def __init__(self):
        self.w: Tensor = None

    def build(self, x):
        self.w = popxl.graph_input(x.shape, x.dtype, "w")
        return self.w + x, self.w + 2 * x


class SubtractSingleWeight(popxl.Module):
    def __init__(self):
        self.w: Tensor = None

    def build(self, x):
        self.w = popxl.graph_input(x.shape, x.dtype, "w")
        return self.w - x


class SubtractMultipleWeight(popxl.Module):
    def __init__(self):
        self.w: Tensor = None

    def build(self, x):
        self.w = popxl.graph_input(x.shape, x.dtype, "w")
        return self.w - x, 2 * self.w - x


class AddSubtractSingle1(popxl.Module):
    def __init__(self):
        self.w: Tensor = None
        self.y: Tensor = None

    def build(self, x):
        self.w = popxl.graph_input(x.shape, x.dtype, "w")
        self.y = popxl.graph_input(x.shape, x.dtype, "y")
        return self.w + x - self.y


class AddSubtractSingle2(popxl.Module):
    def __init__(self):
        self.w: Tensor = None

    def build(self, x, y):
        self.w = popxl.graph_input(x.shape, x.dtype, "w")
        return self.w + x - y


class AddVariableSingle1(popxl.Module):
    def build(self, x, y):
        return x + y


class AddVariableSingle2(popxl.Module):
    def build(self, x, y):
        return x + y


class TestConditional:
    def test_conditional_with_single_output_subgraphs_1(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        na = np.array([20], dtype=np.float32)
        nb = np.array([10], dtype=np.float32)

        with graph:
            input_a = popxl.h2d_stream([1], popxl.float32, name="instream_a")
            input_b = popxl.h2d_stream([1], popxl.float32, name="instream_b")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            ha = ops.host_load(input_a, "a")
            hb = ops.host_load(input_b, "b")
            cond = ops.host_load(input_cond, "cond")
            thenfn = AddSingleConstant()
            elsefn = SubtractSingleConstant()
            graph_then = ir.create_graph(thenfn, ha)
            graph_else = ir.create_graph(elsefn, hb)
            out = ops.conditional(cond=cond,
                                  then_branch=graph_then,
                                  else_branch=graph_else,
                                  then_inputs=[ha],
                                  else_inputs=[hb])[0]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_a:
                    na,
                    input_b:
                    nb,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                expected_output = na + 5 if condition else nb - 3
                assert list(outputs.values())[0] == expected_output

    def test_conditional_with_single_output_subgraphs_2(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        na = np.array([23, 10, 22, 21], dtype=np.float32)
        nb = np.array([10, 18, 32, 12], dtype=np.float32)

        with graph:
            input_a = popxl.h2d_stream([4], popxl.float32, name="instream_a")
            input_b = popxl.h2d_stream([4], popxl.float32, name="instream_b")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            ha = ops.host_load(input_a, "a")
            hb = ops.host_load(input_b, "b")
            cond = ops.host_load(input_cond, "cond")
            thenfn = MultiplySingleVariable()
            elsefn = DivideSingleVariable()
            graph_then = ir.create_graph(thenfn, ha, hb)
            graph_else = ir.create_graph(elsefn, ha, hb)
            out = ops.conditional(cond=cond,
                                  then_branch=graph_then,
                                  else_branch=graph_else,
                                  then_inputs=[ha, hb],
                                  else_inputs=[ha, hb])[0]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_a:
                    na,
                    input_b:
                    nb,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                expected_output = na * nb if condition else na / nb
                np.testing.assert_allclose(expected_output,
                                           list(outputs.values())[0],
                                           rtol=1e-8,
                                           atol=1e-8)

    def test_conditional_with_single_output_subgraphs_3(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        nx = np.array([23, 10, 22, 21], dtype=np.float32)
        nw = np.array([10, 18, 32, 12], dtype=np.float32)

        with graph:
            input_x = popxl.h2d_stream([4], popxl.float32, name="instream_x")
            input_w = popxl.h2d_stream([4], popxl.float32, name="instream_w")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            x = ops.host_load(input_x, "x")
            w = ops.host_load(input_w, "w")
            cond = ops.host_load(input_cond, "cond")
            add_weight0 = AddSingleWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x)
            sub_weight0 = SubtractSingleWeight()
            sub_weight_graph0 = ir.create_graph(sub_weight0, x)
            out = ops.conditional(cond=cond,
                                  then_branch=add_weight_graph0,
                                  else_branch=sub_weight_graph0,
                                  then_inputs=[x],
                                  else_inputs=[x],
                                  then_inputs_dict={add_weight0.w: w},
                                  else_inputs_dict={sub_weight0.w: w})[0]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_x:
                    nx,
                    input_w:
                    nw,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                expected_output = nw + nx if condition else nw - nx
                np.testing.assert_allclose(expected_output,
                                           list(outputs.values())[0],
                                           rtol=1e-8,
                                           atol=1e-8)

    def test_conditional_with_single_output_subgraphs_4(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        nx = np.array([23, 10, 22, 21], dtype=np.float32)
        nw = np.array([10, 18, 32, 12], dtype=np.float32)

        with graph:
            input_x = popxl.h2d_stream([4], popxl.float32, name="instream_x")
            input_w = popxl.h2d_stream([4], popxl.float32, name="instream_w")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            x = ops.host_load(input_x, "x")
            w = ops.host_load(input_w, "w")
            y = 2 * x
            cond = ops.host_load(input_cond, "cond")
            add_weight0 = AddSubtractSingle1()
            add_weight_graph0 = ir.create_graph(add_weight0, x)
            out = ops.conditional(cond=cond,
                                  then_branch=add_weight_graph0,
                                  else_branch=add_weight_graph0,
                                  then_inputs=[x],
                                  else_inputs=[x],
                                  then_inputs_dict={
                                      add_weight0.y: y,
                                      add_weight0.w: w
                                  },
                                  else_inputs_dict={
                                      add_weight0.y: y,
                                      add_weight0.w: w
                                  })[0]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_x:
                    nx,
                    input_w:
                    nw,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                expected_output = nw + nx - 2 * nx
                np.testing.assert_allclose(expected_output,
                                           list(outputs.values())[0],
                                           rtol=1e-8,
                                           atol=1e-8)

    def test_conditional_with_single_output_subgraphs_5(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        nx = np.array([23, 10, 22, 21], dtype=np.float32)
        nw = np.array([10, 18, 32, 12], dtype=np.float32)

        with graph:
            input_x = popxl.h2d_stream([4], popxl.float32, name="instream_x")
            input_w = popxl.h2d_stream([4], popxl.float32, name="instream_w")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            x = ops.host_load(input_x, "x")
            w = ops.host_load(input_w, "w")
            y = 2 * x
            cond = ops.host_load(input_cond, "cond")
            add_weight0 = AddSubtractSingle2()
            add_weight_graph0 = ir.create_graph(add_weight0, x, y)
            out = ops.conditional(cond=cond,
                                  then_branch=add_weight_graph0,
                                  else_branch=add_weight_graph0,
                                  then_inputs=[x],
                                  else_inputs=[x],
                                  then_inputs_dict={
                                      add_weight0.w: w,
                                      add_weight_graph0.inputs[1]: y
                                  },
                                  else_inputs_dict={
                                      add_weight0.w: w,
                                      add_weight_graph0.inputs[1]: y
                                  })[0]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_x:
                    nx,
                    input_w:
                    nw,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                expected_output = nw + nx - 2 * nx
                np.testing.assert_allclose(expected_output,
                                           list(outputs.values())[0],
                                           rtol=1e-8,
                                           atol=1e-8)

    def test_conditional_with_single_output_subgraphs_6(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        nx = np.array([23, 10, 22, 21], dtype=np.float32)

        with graph:
            input_x = popxl.h2d_stream([4], popxl.float32, name="instream_x")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            x = ops.host_load(input_x, "x")
            y = 2 * x
            cond = ops.host_load(input_cond, "cond")
            add_weight1 = AddVariableSingle1()
            add_weight_graph0 = ir.create_graph(add_weight1, x, y)
            add_weight2 = AddVariableSingle2()
            add_weight_graph1 = ir.create_graph(add_weight2, x, y)
            out = ops.conditional(cond=cond,
                                  then_branch=add_weight_graph0,
                                  else_branch=add_weight_graph1,
                                  then_inputs=[x, x],
                                  else_inputs=[y, y])[0]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_x:
                    nx,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                expected_output = nx + nx if condition else 2 * nx + 2 * nx
                np.testing.assert_allclose(expected_output,
                                           list(outputs.values())[0],
                                           rtol=1e-8,
                                           atol=1e-8)

    @pytest.mark.parametrize("out_index", [0, 1])
    def test_conditional_with_multiple_output_subgraphs_1(
            self, out_index: int):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        na = np.array([20], dtype=np.float32)
        nb = np.array([10], dtype=np.float32)

        with graph:
            input_a = popxl.h2d_stream([1], popxl.float32, name="instream_a")
            input_b = popxl.h2d_stream([1], popxl.float32, name="instream_b")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            ha = ops.host_load(input_a, "ha")
            hb = ops.host_load(input_b, "hb")
            cond = ops.host_load(input_cond, "cond")
            thenfn = AddMultipleConstant()
            elsefn = SubtractMultipleConstant()
            graph_then = ir.create_graph(thenfn, ha)
            graph_else = ir.create_graph(elsefn, hb)
            outs = ops.conditional(cond=cond,
                                   then_branch=graph_then,
                                   else_branch=graph_else,
                                   then_inputs=[ha],
                                   else_inputs=[hb])
            out = outs[out_index]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_a:
                    na,
                    input_b:
                    nb,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                res = list(outputs.values())[0]
                if condition and (out_index == 0):
                    assert res == na + 5
                elif condition and (out_index == 1):
                    assert res == na + 10
                elif not condition and (out_index == 0):
                    assert res == nb - 3
                elif not condition and (out_index == 1):
                    assert res == nb - 6

    @pytest.mark.parametrize("out_index", [0, 1])
    def test_conditional_with_multiple_output_subgraphs_2(
            self, out_index: int):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        na = np.array([23, 10, 22, 21], dtype=np.float32)
        nb = np.array([10, 18, 32, 12], dtype=np.float32)

        with graph:
            input_a = popxl.h2d_stream([4], popxl.float32, name="instream_a")
            input_b = popxl.h2d_stream([4], popxl.float32, name="instream_b")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            ha = ops.host_load(input_a, "a")
            hb = ops.host_load(input_b, "b")
            cond = ops.host_load(input_cond, "cond")
            thenfn = MultiplyMultipleVariable()
            elsefn = DivideMultipleVariable()
            graph_then = ir.create_graph(thenfn, ha, hb)
            graph_else = ir.create_graph(elsefn, ha, hb)
            outs = ops.conditional(cond=cond,
                                   then_branch=graph_then,
                                   else_branch=graph_else,
                                   then_inputs=[ha, hb],
                                   else_inputs=[ha, hb])
            out = outs[out_index]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_a:
                    na,
                    input_b:
                    nb,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                res = list(outputs.values())[0]
                if condition and (out_index == 0):
                    np.testing.assert_allclose(na * nb,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)
                elif condition and (out_index == 1):
                    np.testing.assert_allclose(2 * na * nb,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)
                elif not condition and (out_index == 0):
                    np.testing.assert_allclose(na / nb,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)
                elif not condition and (out_index == 1):
                    np.testing.assert_allclose(2 * na / nb,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)

    @pytest.mark.parametrize("out_index", [0, 1])
    def test_conditional_with_multiple_output_subgraphs_3(
            self, out_index: int):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        nx = np.array([23, 10, 22, 21], dtype=np.float32)
        nw = np.array([10, 18, 32, 12], dtype=np.float32)

        with graph:
            input_x = popxl.h2d_stream([4], popxl.float32, name="instream_x")
            input_w = popxl.h2d_stream([4], popxl.float32, name="instream_w")
            input_cond = popxl.h2d_stream([1],
                                          popxl.bool,
                                          name="instream_cond")
            hx = ops.host_load(input_x, "x")
            hw = ops.host_load(input_w, "w")
            cond = ops.host_load(input_cond, "cond")
            add_weight0 = AddMultipleWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, hx)
            sub_weight0 = SubtractMultipleWeight()
            sub_weight_graph0 = ir.create_graph(sub_weight0, hx)
            outs = ops.conditional(cond=cond,
                                   then_branch=add_weight_graph0,
                                   else_branch=sub_weight_graph0,
                                   then_inputs=[hx],
                                   else_inputs=[hx],
                                   then_inputs_dict={add_weight0.w: hw},
                                   else_inputs_dict={sub_weight0.w: hw})
            out = outs[out_index]
            o_d2h = popxl.d2h_stream(out.shape, out.dtype, name="out_stream")
            ops.host_store(o_d2h, out)
        with popxl.Session(ir, "ipu_model") as session:
            for condition in (True, False):
                outputs = session.run({
                    input_x:
                    nx,
                    input_w:
                    nw,
                    input_cond:
                    np.array(condition, dtype=bool)
                })
                res = list(outputs.values())[0]
                if condition and (out_index == 0):
                    np.testing.assert_allclose(nw + nx,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)
                elif condition and (out_index == 1):
                    np.testing.assert_allclose(nw + 2 * nx,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)
                elif not condition and (out_index == 0):
                    np.testing.assert_allclose(nw - nx,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)
                elif not condition and (out_index == 1):
                    np.testing.assert_allclose(2 * nw - nx,
                                               res,
                                               rtol=1e-8,
                                               atol=1e-8)
