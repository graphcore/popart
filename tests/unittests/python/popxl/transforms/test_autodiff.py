# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
import pytest
import numpy as np

import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from popxl.transforms.autodiff import ExpectedConnectionType
from typing import Callable


@pytest.mark.parametrize(
    "gradsRequiredFun",
    [lambda fwd: [fwd.W, fwd.b], lambda fwd: [fwd.b, fwd.W]])
def test_subgraph(gradsRequiredFun):
    class ScaleNShift(popxl.Module):
        def __init__(self):
            self.W: popxl.Tensor = None
            self.b: popxl.Tensor = None

        def build(self, x: popxl.Tensor, out_features: int,
                  bias: bool = True) -> popxl.Tensor:
            self.W = popxl.graph_input((x.shape[-1], out_features),
                                       popxl.float32, "W")
            y = ops.mul(x, self.W)
            if bias:
                self.b = popxl.graph_input((out_features, ), popxl.float32,
                                           "b")
                y = y + self.b
            return y

    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        h2d = popxl.h2d_stream((16, 16), popxl.dtypes.float32)
        x = ops.host_load(h2d, "x")

        W = popxl.variable(np.random.normal(0, 0.1, (16, 16)), name="W")
        b = popxl.variable(np.zeros(16), name="b", dtype=popxl.dtypes.float32)

        ss = ScaleNShift()
        ss_graph = ir.create_graph(ss, x, out_features=16)

        call_info = ops.call_with_info(ss_graph,
                                       x,
                                       inputs_dict={
                                           ss.W: W,
                                           ss.b: b
                                       })

        y = call_info.outputs[0]
        d2h = popxl.d2h_stream(y.shape, y.dtype)
        ops.host_store(d2h, y)

    assert len(ss_graph.inputs) == 3
    assert len(ss_graph.outputs) == 1

    grads_required = gradsRequiredFun(ss)
    ss_bwd_info = popxl.transforms.autodiff(ss_graph,
                                            grads_required=grads_required)

    # Check expected outputs matches gradsRequired and that they are
    # in the same order.
    assert len(ss_bwd_info.expected_outputs) == len(grads_required)
    for exp_out, grad in zip(ss_bwd_info.expected_outputs, grads_required):
        assert exp_out.connection_type == ExpectedConnectionType.FwdGrad
        assert exp_out.fwd_tensor == grad

    # Check an additional output has been added to the fwd graph.
    assert len(ss_graph.outputs) == 2

    bwd_graph = ss_bwd_info.graph

    assert isinstance(bwd_graph, popxl.Graph)

    assert len(ss_bwd_info.expected_inputs) == len(bwd_graph.inputs)
    assert len(ss_bwd_info.expected_outputs) == len(bwd_graph.outputs)

    for op in bwd_graph._pb_graph.getOps():
        grad_ops = (_ir.op.SumOp, _ir.op.MulArg0GradOp, _ir.op.MulArg1GradOp,
                    _ir.op.AddArg0GradOp, _ir.op.AddArg1GradOp)
        assert isinstance(op, grad_ops)

    with main:
        grad_seed = popxl.constant(np.ones((16, 16), np.float32))
        activations = ss_bwd_info.inputs_dict(call_info)
        grads_with_info = ops.call_with_info(bwd_graph,
                                             grad_seed,
                                             inputs_dict=activations)
        grads = grads_with_info.outputs

    assert len(grads) == len(ss_bwd_info.expected_outputs)

    grad_subgraph_tensor_map = ss_bwd_info.fwd_graph_ins_to_grad_parent_outs(
        grads_with_info)
    assert all(t.name in g.name for t, g in grad_subgraph_tensor_map.items())
    assert all(t.shape == g.shape for t, g in grad_subgraph_tensor_map.items())
    assert all(
        'ScaleNShift' in t.id for t, g in grad_subgraph_tensor_map.items())

    grad_tensor_map = ss_bwd_info.fwd_parent_ins_to_grad_parent_outs(
        call_info, grads_with_info)
    assert all(t.shape == g.shape for t, g in grad_tensor_map.items())
    assert all('ScaleNShift' not in t.id for t, g in grad_tensor_map.items())


def test_grad_graph_info_repr():
    def subgraph1(a: popxl.Tensor):
        return a + a

    ir = popxl.Ir()
    with ir.main_graph:
        a = popxl.variable([1], name="bob")
        fwd_graph = ir.create_graph(subgraph1, a.spec)
        fwd_call_info = ops.call_with_info(fwd_graph, a)

        y = fwd_call_info.outputs[0]
        d2h = popxl.d2h_stream(y.shape, y.dtype)
        ops.host_store(d2h, y)

        bwd_info = popxl.transforms.autodiff(fwd_graph)
        bwd_info_repr = repr(bwd_info)
        assert bool(
            re.match(
                r"GradGraphInfo\[(.|\n)*?graph\=(.|\n)*?expected_inputs\=(.|\n)*?expected_outputs\=(.|\n)*?\]",
                bwd_info_repr))


def modifying_subgraph_1(a: popxl.Tensor, b: popxl.Tensor) -> popxl.Tensor:
    # Modifying
    # will raise.
    a += b
    return a


def modifying_subgraph_2(a: popxl.Tensor, b: popxl.Tensor) -> popxl.Tensor:
    # Modifying
    # will raise.
    a = ops.gelu_(a)
    c = a + b
    return c


def modifying_subgraph_3(a: popxl.Tensor, b: popxl.Tensor) -> popxl.Tensor:
    # Inplace but not modifying (concat_ doesn't modify.)
    # will not raise.
    a = ops.concat_([a, b], axis=0)
    return a


def modifying_subgraph_4(a: popxl.Tensor, b: popxl.Tensor) -> popxl.Tensor:
    # Modifying
    # will raise.
    return ops.var_updates.accumulate_(a, b, 0.1)


@pytest.mark.parametrize("subgraph,willraise", [(modifying_subgraph_1, True),
                                                (modifying_subgraph_2, True),
                                                (modifying_subgraph_3, False),
                                                (modifying_subgraph_4, True)])
def test_modifying_ops(subgraph: Callable, willraise: bool):
    """Test that an error is/isn't thrown when trying to autodiff a subgraph with/without
    a modifying op in.

    Args:
        subgraph (Callable): The subgraph to use, see above for definitions.
        willraise (bool): Whether or not the graph created will raise an error.
    """
    import popart

    ir = popxl.Ir()
    with ir.main_graph:
        a = popxl.variable([1], name="a")
        b = popxl.variable([1], name="b")
        fwd_graph = ir.create_graph(subgraph, a.spec, b.spec)
        fwd_call_info = ops.call_with_info(fwd_graph, a, b)

        y = fwd_call_info.outputs[0]
        d2h = popxl.d2h_stream(y.shape, y.dtype)
        ops.host_store(d2h, y)

        if willraise:
            with pytest.raises(popart.popart_exception) as e_info:
                _ = popxl.transforms.autodiff(fwd_graph)

            assert (e_info.value.args[0].startswith(
                f"The graph {subgraph.__name__}_subgraph(0) contains a modifying op"
            ))
        else:
            _ = popxl.transforms.autodiff(fwd_graph)
