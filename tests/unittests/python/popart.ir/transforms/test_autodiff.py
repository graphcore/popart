# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
import pytest
import numpy as np

import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from popart.ir.transforms.autodiff import ExpectedConnectionType


@pytest.mark.parametrize(
    "gradsRequiredFun",
    [lambda fwd: [fwd.W, fwd.b], lambda fwd: [fwd.b, fwd.W]])
def test_subgraph(gradsRequiredFun):
    class ScaleNShift(pir.Module):
        def __init__(self):
            self.W: pir.Tensor = None
            self.b: pir.Tensor = None

        def build(self, x: pir.Tensor, out_features: int,
                  bias: bool = True) -> pir.Tensor:
            self.W = pir.subgraph_input((x.shape[-1], out_features),
                                        pir.float32, "W")
            y = ops.mul(x, self.W)
            if bias:
                self.b = pir.subgraph_input((out_features, ), pir.float32, "b")
                y = y + self.b
            return y

    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        h2d = pir.h2d_stream((16, 16), pir.dtypes.float32)
        x = ops.host_load(h2d, "x")

        W = pir.variable(np.random.normal(0, 0.1, (16, 16)), name="W")
        b = pir.variable(np.zeros(16), name="b", dtype=pir.dtypes.float32)

        ss = ScaleNShift()
        ss_graph = ir.create_graph(ss, x, out_features=16)

        call_info = ops.call_with_info(ss_graph,
                                       x,
                                       subgraph_in_to_parent_in={
                                           ss.W: W,
                                           ss.b: b
                                       })

        y = call_info.get_output_tensors()[0]
        d2h = pir.d2h_stream(y.shape, y.dtype)
        ops.host_store(d2h, y)

    assert len(ss_graph.get_input_tensors()) == 3
    assert len(ss_graph.get_output_tensors()) == 1

    grads_required = gradsRequiredFun(ss)
    ss_bwd_info = pir.transforms.autodiff(ss_graph,
                                          grads_required=grads_required)

    # Check expected outputs matches gradsRequired and that they are
    # in the same order.
    assert len(ss_bwd_info.expected_outputs) == len(grads_required)
    for exp_out, grad in zip(ss_bwd_info.expected_outputs, grads_required):
        assert exp_out.connection_type == ExpectedConnectionType.FwdGrad
        assert exp_out.fwd_tensor == grad

    # Check an additional output has been added to the fwd graph.
    assert len(ss_graph.get_output_tensors()) == 2

    bwd_graph = ss_bwd_info.graph

    assert isinstance(bwd_graph, pir.Graph)

    assert len(ss_bwd_info.expected_inputs) == len(
        bwd_graph.get_input_tensors())
    assert len(ss_bwd_info.expected_outputs) == len(
        bwd_graph.get_output_tensors())

    for op in bwd_graph._pb_graph.getOps():
        grad_ops = (_ir.op.SumOp, _ir.op.MulArg0GradOp, _ir.op.MulArg1GradOp,
                    _ir.op.AddArg0GradOp, _ir.op.AddArg1GradOp)
        assert isinstance(op, grad_ops)

    with main:
        grad_seed = pir.constant(np.ones((16, 16), np.float32))
        activations = ss_bwd_info.get_inputs_from_forward_call_info(call_info)
        grads_with_info = ops.call_with_info(
            bwd_graph, grad_seed, subgraph_in_to_parent_in=activations)
        grads = grads_with_info.get_output_tensors()

    assert len(grads) == len(ss_bwd_info.expected_outputs)

    grad_subgraph_tensor_map = ss_bwd_info.get_fwd_subgraph_to_grad_tensor_map(
        grads_with_info)
    assert all(t.name in g.name for t, g in grad_subgraph_tensor_map.items())
    assert all(t.shape == g.shape for t, g in grad_subgraph_tensor_map.items())
    assert all(
        'ScaleNShift' in t.id for t, g in grad_subgraph_tensor_map.items())

    grad_tensor_map = ss_bwd_info.get_fwd_inputs_to_grad_tensor_map(
        call_info, grads_with_info)
    assert all(t.shape == g.shape for t, g in grad_tensor_map.items())
    assert all('ScaleNShift' not in t.id for t, g in grad_tensor_map.items())


def test_grad_graph_info_repr():
    def subgraph1(a: pir.Tensor):
        return a + a

    ir = pir.Ir()
    with ir.main_graph():
        a = pir.variable([1], name="bob")
        fwd_graph = ir.create_graph(subgraph1, a.spec)
        fwd_call_info = ops.call_with_info(fwd_graph, a)

        y = fwd_call_info.get_output_tensors()[0]
        d2h = pir.d2h_stream(y.shape, y.dtype)
        ops.host_store(d2h, y)

        bwd_info = pir.transforms.autodiff(fwd_graph)
        bwd_info_repr = repr(bwd_info)
        assert bool(
            re.match(
                r"GradGraphInfo\[(.|\n)*?graph\=(.|\n)*?expected_inputs\=(.|\n)*?expected_outputs\=(.|\n)*?\]",
                bwd_info_repr))
