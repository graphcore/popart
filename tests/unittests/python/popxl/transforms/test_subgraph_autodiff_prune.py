# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops


def test_autodiff_prune():

    ir = popxl.Ir()
    main = ir.main_graph

    def fwd(x, gamma, beta):
        return ops.group_norm(x, gamma, beta, 1)

    with main:
        x = popxl.variable(np.random.normal(0, 0.02, (4, 16)), popxl.float32)
        gamma = popxl.variable(np.ones((16,)), popxl.float32)
        beta = popxl.variable(np.zeros((16,)), popxl.float32)

        fwd_graph = ir.create_graph(fwd, x.spec, gamma.spec, beta.spec)
        grad_info = popxl.transforms.autodiff(
            fwd_graph, grads_required=fwd_graph.inputs[:1]
        )

        fwd_info = ops.call_with_info(fwd_graph, x, gamma, beta)
        _ = ops.call(
            grad_info.graph,
            popxl.constant(np.ones((4, 16)), popxl.float32),
            inputs_dict=grad_info.inputs_dict(fwd_info),
        )

    grad_ops = grad_info.graph._pb_graph.getOps()
    group_norm_grad = None
    for op in grad_ops:
        if op.opType() == "GroupNormalizationGrad":
            group_norm_grad = op
            break
    assert group_norm_grad

    # GroupNormalizationGrad will have all of its outputs even if not all
    # lead to required grads, because it is invalid for an Op to not have
    # all of its outputs connected.
    # The backward graph will still only output the required grads.

    assert len(group_norm_grad.getOutputTensors()) == 3
    assert (
        len(grad_info.expected_outputs)
        == len(grad_info.graph._pb_graph.getOutputIds())
        == 1
    )
