# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np

import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops


def test_subgraph():
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

    ss_bwd_info = pir.transforms.autodiff.autodiff(ss_graph)

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
        activations = pir.transforms.autodiff.get_expected_forward_inputs_from_call(
            call_info, ss_bwd_info)
        grads = ops.call(bwd_graph,
                         grad_seed,
                         subgraph_in_to_parent_in=activations)

    assert len(grads) == len(ss_bwd_info.expected_outputs)
