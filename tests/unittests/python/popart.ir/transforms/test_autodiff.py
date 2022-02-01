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

    ss_bwd_info = pir.transforms.autodiff(ss_graph)

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


def test_non_required_grads():
    """This is a test to show that if you don't require the gradients for `gamma` and `beta`,
        the output of the autodiff subgraph shouldn't include them.

    However previously, autodiff didn't even add them to the Grad Op, causing a map::at error."""
    import popart

    ir = pir.Ir()
    main = ir.main_graph()

    def fwd(x, gamma, beta):
        return ops.group_norm(x, gamma, beta, 1)

    with main:
        x = pir.variable(np.random.normal(0, 0.02, (4, 16)), pir.float32)
        gamma = pir.variable(np.ones((16, )), pir.float32, name="gamma")
        beta = pir.variable(np.zeros((16, )), pir.float32, name="beta")

        fwd_graph = ir.create_graph(fwd, x, gamma, beta)
        grads_required = fwd_graph.get_input_tensors()[:1]
        # Note only 1 tensor here.
        print(grads_required)
        grad_info = pir.transforms.autodiff(fwd_graph,
                                            grads_required=grads_required)

        fwd_info = ops.call_with_info(fwd_graph, x, gamma, beta)
        dx = ops.call(grad_info.graph,
                      pir.constant(np.ones((4, 16)), pir.float32),
                      subgraph_in_to_parent_in=grad_info.
                      get_inputs_from_forward_call_info(fwd_info))

    grad_ops = grad_info.graph._pb_graph.getOps()
    group_norm_grad = None
    for op in grad_ops:
        if op.opType() == "GroupNormalizationGrad":
            group_norm_grad = op
            break
    assert group_norm_grad

    dataFlow = popart.DataFlow(batchesPerStep=1, anchorTensors={})

    _ir = ir._pb_ir
    _ir.setDataFlow(dataFlow)

    opts = ir._pb_ir.getSessionOptions()
    opts.constantWeights = False
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    _ir.removeIsolatedGraphs()
    _ir.removeIsolatedTensors(True)

    for g in _ir.getAllGraphs():
        _ir.applyPreAliasPatterns(g)

    _ir.updateVertices()
    _ir.logIr()

    device = popart.DeviceManager().createIpuModelDevice({"numIPUs": 1})

    session = popart.InferenceSession.fromIr(ir=_ir, deviceInfo=device)
    session.prepareDevice()

    # This grad op should have 3 outputs, not 1.
    assert len(group_norm_grad.getOutputTensors()) == 3
