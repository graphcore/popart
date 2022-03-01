# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Milestone 2: Autodiff on Subgraph"""

import popxl
import popxl.ops as ops

from popxl.transforms.autodiff import ExpectedConnection, ExpectedConnectionType
import numpy as np

from typing import Tuple

_BATCH_SIZE = 2
_IN_FEATURES = 8
_OUT_FEATURES = 4

_IN_SHAPE = (_BATCH_SIZE, _IN_FEATURES)
_WEIGHT_SHAPE = (_IN_FEATURES, _OUT_FEATURES)
_BIAS_SHAPE = (_OUT_FEATURES, )
_OUT_SHAPE = (_BATCH_SIZE, _OUT_FEATURES)


class Linear(popxl.Module):
    def __init__(self):
        self.W: popxl.Tensor = None
        self.b: popxl.Tensor = None

    def build(self, x: popxl.Tensor, out_features: int,
              bias: bool = True) -> popxl.Tensor:
        self.W = popxl.graph_input((x.shape[-1], out_features), popxl.float32,
                                   "W")
        y = x @ self.W
        if bias:
            self.b = popxl.graph_input((out_features, ), popxl.float32, "b")
            y = y + self.b
        return y


# Build model using popxl API, then return the underlying Ir.
# Also returns the streams for the input and output tensors, and the data of the
# variables.
def build_model(
) -> Tuple[popxl.Ir, popxl.HostToDeviceStream, popxl.DeviceToHostStream, popxl.
           DeviceToHostStream, popxl.DeviceToHostStream, popxl.
           DeviceToHostStream, np.ndarray, np.ndarray]:
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        x_h2d = popxl.h2d_stream(_IN_SHAPE, popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        W_data = np.random.normal(0, 0.1, _WEIGHT_SHAPE).astype(np.float32)
        b_data = np.zeros(_BIAS_SHAPE, dtype=np.float32)

        W = popxl.variable(W_data, name="W")
        b = popxl.variable(b_data, name="b")

        lin = Linear()
        lin_graph = ir.create_graph(lin, x, out_features=_OUT_FEATURES)

        lin_call_info = ops.call_with_info(lin_graph,
                                           x,
                                           inputs_dict={
                                               lin.W: W,
                                               lin.b: b
                                           })
        y = lin_call_info.outputs[0]

        assert y.shape == _OUT_SHAPE

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="x_stream")
        ops.host_store(y_d2h, y)

    lin_bwd_info = popxl.transforms.autodiff(lin_graph)
    lin_bwd_graph = lin_bwd_info.graph

    with main:
        grad_seed = popxl.constant(np.ones(_OUT_SHAPE, np.float32))
        tensors_required_for_bwd = lin_bwd_info.inputs_dict(lin_call_info)
        lin_bwd_call_info = ops.call_with_info(
            lin_bwd_graph, grad_seed, inputs_dict=tensors_required_for_bwd)

    ##### Extract parent graph x_grad, W_grad, b_grad

    expected_outputs = lin_bwd_info.expected_outputs
    x_grad, W_grad, b_grad = None, None, None

    sg_x = lin_call_info.parent_to_graph(x)
    sg_W = lin_call_info.parent_to_graph(W)
    sg_b = lin_call_info.parent_to_graph(b)

    def get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
            ec: ExpectedConnection) -> popxl.Tensor:
        # If (t, FwdGrad) appears at index i in expected_outputs, it is
        # guaranteed that t' (the grad of t) appears at output index i in the
        # grad graph.
        sg_out_idx = expected_outputs.index(ec)
        op_out_idx = lin_bwd_call_info.graph_to_parent_input_index(sg_out_idx)
        parent_grad = lin_bwd_call_info.parent_output(op_out_idx)

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
            grad: popxl.Tensor) -> popxl.DeviceToHostStream:
        with main:
            d2h = popxl.d2h_stream(grad.shape,
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

    return ir, x_h2d, y_d2h, x_grad_d2h, W_grad_d2h, b_grad_d2h, W_data, b_data


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

    ir.num_host_transfers = 1

    session = popxl.Session(ir, 'ipu_model')

    # Create data for input x
    x_data = np.ones(_IN_SHAPE, dtype=np.float32)
    inputs = {x_h2d: x_data}

    outputs = session.run(inputs)

    def check_tensors(a: np.ndarray, b: np.ndarray):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert np.allclose(a, b, atol=1e-8)

    expected_y = np.matmul(x_data, W_data) + b_data
    check_tensors(outputs[y_d2h], expected_y)

    expected_y_grad = np.ones(_OUT_SHAPE, dtype=np.float32)

    expected_W_grad = x_data.T @ expected_y_grad
    check_tensors(outputs[W_grad_d2h], expected_W_grad)

    expected_x_grad = expected_y_grad @ W_data.T
    check_tensors(outputs[x_grad_d2h], expected_x_grad)

    expected_b_grad = expected_y_grad.sum(axis=0)
    check_tensors(outputs[b_grad_d2h], expected_b_grad)
