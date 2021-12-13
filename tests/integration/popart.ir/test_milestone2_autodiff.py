# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Milestone 2: Autodiff on Subgraph"""

import popart.ir as pir
import popart.ir.ops as ops

import popart._internal.ir as _ir

import popart
from popart.ir.transforms.autodiff import ExpectedConnection, ExpectedConnectionType
import numpy as np

from typing import Tuple

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

_BATCH_SIZE = 2
_IN_FEATURES = 8
_OUT_FEATURES = 4

_IN_SHAPE = (_BATCH_SIZE, _IN_FEATURES)
_WEIGHT_SHAPE = (_IN_FEATURES, _OUT_FEATURES)
_BIAS_SHAPE = (_OUT_FEATURES, )
_OUT_SHAPE = (_BATCH_SIZE, _OUT_FEATURES)


class Linear(pir.Module):
    def __init__(self):
        self.W: pir.Tensor = None
        self.b: pir.Tensor = None

    def build(self, x: pir.Tensor, out_features: int,
              bias: bool = True) -> pir.Tensor:
        self.W = pir.subgraph_input((x.shape[-1], out_features), pir.float32,
                                    "W")
        y = x @ self.W
        if bias:
            self.b = pir.subgraph_input((out_features, ), pir.float32, "b")
            y = y + self.b
        return y


# Build model using popart.ir API, then return the underlying Ir.
# Also returns the streams for the input and output tensors, and the data of the
# variables.
def build_model(
) -> Tuple[_ir.Ir, pir.HostToDeviceStream, pir.DeviceToHostStream, pir.
           DeviceToHostStream, pir.DeviceToHostStream, pir.
           DeviceToHostStream, np.ndarray, np.ndarray]:
    ir = pir.Ir()

    main = ir.main_graph()
    with main:
        x_h2d = pir.h2d_stream(_IN_SHAPE, pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        W_data = np.random.normal(0, 0.1, _WEIGHT_SHAPE).astype(np.float32)
        b_data = np.zeros(_BIAS_SHAPE, dtype=np.float32)

        W = pir.variable(W_data, name="W")
        b = pir.variable(b_data, name="b")

        lin = Linear()
        lin_graph = ir.create_graph(lin, x, out_features=_OUT_FEATURES)

        lin_call_info = ops.call_with_info(lin_graph,
                                           x,
                                           subgraph_in_to_parent_in={
                                               lin.W: W,
                                               lin.b: b
                                           })
        y = lin_call_info.get_output_tensors()[0]

        assert y.shape == _OUT_SHAPE

        y_d2h = pir.d2h_stream(y.shape, y.dtype, name="x_stream")
        ops.host_store(y_d2h, y)

    lin_bwd_info = pir.transforms.autodiff(lin_graph)
    lin_bwd_graph = lin_bwd_info.graph

    with main:
        grad_seed = pir.constant(np.ones(_OUT_SHAPE, np.float32))
        tensors_required_for_bwd = lin_bwd_info.get_inputs_from_forward_call_info(
            lin_call_info)
        lin_bwd_call_info = ops.call_with_info(
            lin_bwd_graph,
            grad_seed,
            subgraph_in_to_parent_in=tensors_required_for_bwd)

    ##### Extract parent graph x_grad, W_grad, b_grad

    expected_outputs = lin_bwd_info.expected_outputs
    x_grad, W_grad, b_grad = None, None, None

    sg_x = lin_call_info.op_in_to_subgraph_in_tensor(x)
    sg_W = lin_call_info.op_in_to_subgraph_in_tensor(W)
    sg_b = lin_call_info.op_in_to_subgraph_in_tensor(b)

    def get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
            ec: ExpectedConnection) -> pir.Tensor:
        # If (t, FwdGrad) appears at index i in expected_outputs, it is
        # guaranteed that tâ (the grad of t) appears at output index i in the
        # grad graph.
        sg_out_idx = expected_outputs.index(ec)
        op_out_idx = lin_bwd_call_info.subgraph_in_to_op_in_index(sg_out_idx)
        parent_grad = lin_bwd_call_info.get_op_output_tensor(op_out_idx)

        return parent_grad

    for ec in expected_outputs:
        # Should always be the case for expected_outputs
        assert ec.connection_type == ExpectedConnectionType.FwdGrad

        sg_fwd_tensor = ec.fwd_tensor

        if sg_fwd_tensor == sg_x:
            x_grad = get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
                ec)
        elif sg_fwd_tensor == sg_W:
            W_grad = get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
                ec)
        elif sg_fwd_tensor == sg_b:
            b_grad = get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
                ec)

    assert x_grad is not None
    assert W_grad is not None
    assert b_grad is not None

    # HostStore grads and collect d2h streams
    def host_store_and_return_d2h_stream(
            grad: pir.Tensor) -> pir.DeviceToHostStream:
        with main:
            d2h = pir.d2h_stream(grad.shape,
                                 grad.dtype,
                                 name=grad.name + "_stream")
            ops.host_store(d2h, grad)
        return d2h

    x_grad_d2h = host_store_and_return_d2h_stream(x_grad)
    W_grad_d2h = host_store_and_return_d2h_stream(W_grad)
    b_grad_d2h = host_store_and_return_d2h_stream(b_grad)

    assert x_grad_d2h is not None
    assert W_grad_d2h is not None
    assert b_grad_d2h is not None

    return ir._pb_ir, x_h2d, y_d2h, x_grad_d2h, W_grad_d2h, b_grad_d2h, W_data, b_data


def test_autodiff():
    """
    Builds and executes the following (pseudocode):
    x = ones(shape=(b, m))
    W = random_normal(shape=(m, n))
    b = zeros(shape=(n,))
    y = x @ W + b
    y.backward()

    Then compute in numpy:
    y, y_grad, W_grad, x_grad, b_grad
    """
    ir, x_h2d, y_d2h, x_grad_d2h, W_grad_d2h, b_grad_d2h, W_data, b_data = build_model(
    )

    x_id = x_h2d.tensor_id()
    y_id = y_d2h.tensor_id()
    x_grad_id = x_grad_d2h.tensor_id()
    W_grad_id = W_grad_d2h.tensor_id()
    b_grad_id = b_grad_d2h.tensor_id()

    ids_to_anchor = [
        y_id,
        x_grad_id,
        W_grad_id,
        b_grad_id,
    ]

    arts = {}

    art_all = popart.AnchorReturnType("All")
    for id in ids_to_anchor:
        arts[id] = art_all

    bps = 1
    dataFlow = popart.DataFlow(bps, arts)
    ir.setDataFlow(dataFlow)

    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    ir.updateVertices()

    ir.setPatterns(_ir.patterns.Patterns(_ir.patterns.PatternsLevel.Minimal))
    for g in ir.getAllGraphs():
        ir.applyPreAliasPatterns(g)
        ir.applyInplacePattern(g)
    ir.updateVertices()

    session = popart.InferenceSession.fromIr(
        ir=ir, deviceInfo=tu.create_test_device())

    session.prepareDevice()

    # Create data for input x
    x_data = np.ones(_IN_SHAPE, dtype=np.float32)

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    stepio = popart.PyStepIO({x_id: x_data}, anchors)

    session.weightsFromHost()
    session.run(stepio)

    def check_tensors(a: np.ndarray, b: np.ndarray):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert np.allclose(a, b, atol=1e-8)

    expected_y = np.matmul(x_data, W_data) + b_data
    check_tensors(anchors[y_id], expected_y)

    expected_y_grad = np.ones(_OUT_SHAPE, dtype=np.float32)

    expected_W_grad = x_data.T @ expected_y_grad
    check_tensors(anchors[W_grad_id], expected_W_grad)

    expected_x_grad = expected_y_grad @ W_data.T
    check_tensors(anchors[x_grad_id], expected_x_grad)

    expected_b_grad = expected_y_grad.sum(axis=0)
    check_tensors(anchors[b_grad_id], expected_b_grad)
