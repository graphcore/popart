# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from utils import contains_op_of_type, num_op_of_type

import popart
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from popxl.tensor import graph_input, Tensor, TensorByRef

# `import test_util` requires adding to sys.path
sys.path.append(
    str(
        Path(__file__).resolve().parent.parent.parent.parent.parent /
        "integration"))
import test_util as tu


class AddOne(popxl.Module):
    def __init__(self):
        pass

    def build(self, x: Tensor) -> Tensor:
        return x + 1


class Linear(popxl.Module):
    def __init__(self):
        self.W: Tensor = None
        self.b: Tensor = None

    def build(self, x: Tensor, out_features: int,
              bias: bool = True) -> Tuple[Tensor, ...]:
        self.W = graph_input((x.shape[-1], out_features), popxl.float32, "W")
        y = x @ self.W
        if bias:
            self.b = graph_input((out_features, ), popxl.float32, "b")
            y = y + self.b
        # TODO: T49312 Add a helper that returns tensors from a subgraph as a mapping
        return y


class LinearHostLoad(popxl.Module):
    def __init__(self):
        self.W: Tensor = None
        self.b: Tensor = None

        self.h2d = popxl.h2d_stream((16, 16), popxl.float32, name="x_stream")
        self.d2h = popxl.d2h_stream((16, 16), popxl.float32, name="y_stream")

    def build(self, x: Tensor, out_features: int,
              bias: bool = True) -> Tuple[Tensor, ...]:
        x = ops.host_load(self.h2d, "x")
        self.W = graph_input((x.shape[-1], out_features), popxl.float32, "W")
        y = x @ self.W
        if bias:
            self.b = graph_input((out_features, ), popxl.float32, "b")
            y = y + self.b
        ops.host_store(self.d2h, y)
        return y


def run_ir(ir: popxl.Ir, bps: int, y_id: str,
           inputs: Dict[str, np.array]) -> np.ndarray:
    """Take the given ir and inputs and run it.

    Args:
        ir (popxl.Ir): The ir to run.
        bps (int): Batches per step.
        y_id (str): The output id, will return this anchor.
        inputs (Dict[str, np.array]): Dict of the inputs to pass to PyStepIO.

    Returns:
        np.array: The anchor associated with y_id.
    """
    _pb_ir = ir._pb_ir  # Internal ir

    dataFlow = popart.DataFlow(
        batchesPerStep=bps,
        anchorTensors={y_id: popart.AnchorReturnType("All")})
    _pb_ir.setDataFlow(dataFlow)

    opts = _pb_ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    _pb_ir.updateVertices()

    _pb_ir.setPatterns(
        _ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))

    with tu.create_test_device() as device:
        session = popart.InferenceSession.fromIr(ir=_pb_ir, deviceInfo=device)

        _pb_ir.logIr()

        session.prepareDevice()

        # Create buffers for anchors
        anchors = session.initAnchorArrays()

        # Run the model
        stepio = popart.PyStepIO(inputs=inputs, outputs=anchors)

        session.weightsFromHost()
        session.run(stepio)

    y = anchors[y_id]
    return y


class TestRepeat:
    @pytest.mark.parametrize("repeat_count", [4, 10])
    def test_repeat_fn(self, repeat_count: int):
        ir = popxl.Ir()
        g = ir.main_graph

        def id_fn(x: Tensor):
            return x

        with g:
            a = popxl.variable(1)

            id_graph = ir.create_graph(id_fn, a)
            b, = ops.repeat(id_graph, repeat_count, a)

        # 2 tensors
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Loop", _ir.op.LoopOp, g)

        assert len(id_graph.tensors) == 1
        assert len(id_graph.variables) == 0
        # Rudimentarily test subgraph has no ops with negative tests
        assert not contains_op_of_type("Loop", _ir.op.LoopOp, id_graph)
        assert not contains_op_of_type("Add", _ir.op.AddOp, id_graph)

    def test_repeat_module_with_internal_inputs_and_multiple_callsites(self):
        """This is the same as the call op's test
            `test_call_module_with_internal_inputs_and_multiple_callsites`,
            with repeats in place of call ops.
        """
        ir = popxl.Ir()
        g = ir.main_graph
        repeat_count: int = 4

        class AddWeight(popxl.Module):
            def __init__(self):
                self.w: Tensor = None

            def build(self, x):
                self.w = popxl.graph_input(x.shape, x.dtype, "w")
                return self.w + x

        with g:
            w0 = popxl.variable(1, name="w0")
            x0 = popxl.variable(1, name="x0")

            # First graph
            add_weight0 = AddWeight()
            add_weight_graph0 = ir.create_graph(add_weight0, x0)

            # First call site
            ops.repeat(add_weight_graph0,
                       repeat_count,
                       x0,
                       inputs_dict={add_weight0.w: w0})

            # Second call site of same graph
            w1 = popxl.variable(1, name="w1")
            x1 = popxl.variable(1, name="x1")

            ops.repeat(add_weight_graph0,
                       repeat_count,
                       x1,
                       inputs_dict={add_weight0.w: w1})

            # Second graph from new instance of module.
            # ir.create_graph should be able to create a new unique Graph name.
            add_weight1 = AddWeight()
            add_weight_graph1 = ir.create_graph(add_weight1, x0)

            # Call second graph. Reuse x0 and w1 as inputs.
            ops.repeat(add_weight_graph1,
                       repeat_count,
                       x0,
                       inputs_dict={add_weight1.w: w1})

            # Third graph that reuses module add_weight1.
            # This calls `build` again, and thus simply overwrites add_weight1.w
            # to be the tensor in the new subgraph add_weight_graph2.
            old_w1_id = add_weight1.w.id
            add_weight_graph2 = ir.create_graph(add_weight1, x1)

            assert old_w1_id != add_weight1.w.id

            # Call third graph. Reuse x1 and w0 as inputs.
            ops.repeat(add_weight_graph2,
                       repeat_count,
                       x1,
                       inputs_dict={add_weight1.w: w0})

        # Test main graph
        # 4 vars + y0 + y1 + y2 + y3.
        # 4 call sites total
        assert len(g.tensors) == 8
        assert len(g.variables) == 4
        assert num_op_of_type("Loop", _ir.op.LoopOp, g) == 4

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
            assert not contains_op_of_type("Loop", _ir.op.LoopOp,
                                           add_weight_subgraph)
            assert not contains_op_of_type("Mul", _ir.op.MulOp,
                                           add_weight_subgraph)

        test_subgraph(add_weight_graph0)
        test_subgraph(add_weight_graph1)
        test_subgraph(add_weight_graph2)

    @pytest.mark.parametrize("repeat_count", [0, 1, 4, 10])
    def test_repeat_simple_addition(self, repeat_count: int):
        """Test that a simple x = x + 1 repeated `repeat_count` times will
        produce x = `repeat_count`

        Args:
            repeat_count (int): Number of times to repeat.
        """
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            one = popxl.constant(0, popxl.dtypes.int32)

            add_one = AddOne()
            add_one_graph = ir.create_graph(add_one, one)

            y, = ops.repeat(add_one_graph, repeat_count, one, inputs_dict={})

            d2h = popxl.d2h_stream(y.shape,
                                   popxl.dtypes.int32,
                                   name="y_stream")
            ops.host_store(d2h, y)

        r_y = run_ir(ir, 1, d2h.tensor_id, {})

        assert r_y == repeat_count

    @pytest.mark.parametrize("repeat_count", [4, 10])
    def test_repeat_subgraph(self, repeat_count: int):
        """Do:
            host load x
            x = (x * W) + b <-- repeat `repeat_count` times.
            host store x

        Args:
            repeat_count (int): How many times to repeat.
        """
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            x_h2d = popxl.h2d_stream((16, 16), popxl.float32, name="x_stream")
            x = ops.host_load(x_h2d, "x")

            W_data = np.random.normal(0, 1, (16, 16)).astype(np.float32)
            W = popxl.variable(W_data, name="W")
            b_data = np.random.normal(0, 0.4, (16)).astype(np.float32)
            b = popxl.variable(b_data, name="b")

            linear = Linear()
            linear_graph = ir.create_graph(linear, x, out_features=16)

            y, = ops.repeat(linear_graph,
                            repeat_count,
                            x,
                            inputs_dict={
                                linear.W: W,
                                linear.b: b
                            })
            y_d2h = popxl.d2h_stream((16, 16), popxl.float32, name="y_stream")
            ops.host_store(y_d2h, y)

        data = np.random.random((16, 16)).astype(np.float32)
        r_y = run_ir(ir,
                     bps=1,
                     y_id=y_d2h.tensor_id,
                     inputs={x_h2d.tensor_id: data})
        out = data
        for _ in range(repeat_count):
            out = np.matmul(out, W_data, dtype=np.float32) + b_data

        assert r_y.shape == out.shape
        # Multiple matmuls mean the values get pretty large and fp differences are common. Hence
        # the large rtol and atol.
        assert np.allclose(r_y, out, rtol=1e-04, atol=1e-03)

    @pytest.mark.parametrize("repeat_count", [4, 10])
    def test_repeat_subgraph_2(self, repeat_count: int):
        """This a repeat in the style of an explicit forward pass loop, where one batch is processed
        per loop iteration. Each loop iteration does:
            host load x       }
            x = (x * W) + b   }--- repeat `repeat_count` times
            host store x      }
        Note this is different to the previous test, in that both the host load and store are within
        the loop body, so new data is loaded and stored each time.

        Args:
            repeat_count (int): How many times to repeat.
        """
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            W_data = np.random.normal(0, 0.1, (16, 16)).astype(np.float32)
            W = popxl.variable(W_data, name="W")
            b_data = np.random.normal(0, 0.4, (16)).astype(np.float32)
            b = popxl.variable(b_data, name="b")

            # This is the loop carried input.
            x = ops.init([16, 16], popxl.float32, "init")

            linear = LinearHostLoad()
            linear_graph = ir.create_graph(linear, x, out_features=16)

            y, = ops.repeat(linear_graph,
                            repeat_count,
                            x,
                            inputs_dict={
                                linear.W: W,
                                linear.b: b
                            })

        data = np.random.random((repeat_count, 16, 16)).astype(np.float32)
        r_y = run_ir(ir,
                     bps=repeat_count,
                     y_id=linear.d2h.tensor_id,
                     inputs={linear.h2d.tensor_id: data})

        for i in range(repeat_count):
            print(f"Batch: {i}")
            out = np.matmul(data[i, :, :], W_data, dtype=np.float32) + b_data
            assert out.shape == r_y[i, :, :].shape
            assert np.allclose(r_y[i, :, :], out, rtol=1e-07, atol=1e-06)

    @pytest.mark.parametrize("repeat_count", [-10, -1])
    def test_repeat_error(self, repeat_count: int):
        """Test an error is thrown with incorrect repeat_count

        Args:
            repeat_count (int): Number of times to repeat.
        """
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            h2d = popxl.h2d_stream((2, 16), popxl.dtypes.float32)
            x = ops.host_load(h2d, "x")

            W = popxl.variable(np.random.normal(0, 0.1, (16, 16)), name="W")
            b = popxl.variable(np.zeros(16), name="b")

            linear = Linear()
            linear_graph = ir.create_graph(linear, x, out_features=16)

            with pytest.raises(ValueError) as e_info:
                y, = ops.repeat(linear_graph,
                                repeat_count,
                                x,
                                inputs_dict={
                                    linear.W: W,
                                    linear.b: b
                                })
            assert e_info.value.args[0].startswith("Repeat count must be >= 0")

    def test_repeat_io_error(self):
        """Test an error is thrown when len(inputs) < len(outputs)"""
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            x = popxl.variable(1)

            def fn(t):
                return t + 1, t + 2

            graph = ir.create_graph(fn, x)

            with pytest.raises(ValueError) as e_info:
                y, = ops.repeat(graph, 10, x)
            assert e_info.value.args[0].startswith("To repeat the subgraph")

    def test_not_by_ref(self):
        ir = popxl.Ir()

        def foo(x: Tensor, y: Tensor):
            return ops.var_updates.accumulate_(x, y), y  # <- modifying op

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)

            g = ir.create_graph(foo, v1, v2)
            loop_info = ops.repeat_with_info(g, 10, v1, v2)

        assert len(g._by_ref_inputs) == 0
        assert not loop_info._op.modifiesIndex(
            2)  # Offset by 1 due to count and keep_going
        assert not loop_info._op.modifiesIndex(3)

    def test_repeat_by_ref_implicit(self):
        ir = popxl.Ir()

        def foo(x: Tensor, y: TensorByRef):
            ops.var_updates.accumulate_(y, x)  # <- modifying op
            return x

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)

            g = ir.create_graph(foo, v1, v2)
            loop_info = ops.repeat_with_info(g, 10, v1, v2)

        assert len(g._by_ref_inputs) == 1
        # Offset by 2 due to count and keep_going
        assert not loop_info._op.modifiesIndex(2)
        assert loop_info._op.modifiesIndex(3)

    def test_repeat_by_ref_explicit(self):
        ir = popxl.Ir()

        def foo(x: TensorByRef, y: Tensor):
            return x + y  # <- non-modifying op

        with ir.main_graph:
            v1 = popxl.variable(1)
            v2 = popxl.variable(2)

            g = ir.create_graph(foo, v1, v2)
            loop_info = ops.repeat_with_info(g, 10, v1, v2)

        assert len(g._by_ref_inputs) == 1
        # Offset by 2 due to count and keep_going
        assert loop_info._op.modifiesIndex(2)
        assert not loop_info._op.modifiesIndex(3)
