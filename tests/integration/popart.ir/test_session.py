# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Tuple

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.tensor import Tensor, graph_input


def test_session():
    w_input = [2.]
    c_input = [1.]

    ir = pir.Ir()
    with ir.main_graph, pir.in_sequence():
        w = pir.variable(w_input, pir.dtypes.float32)
        x_h2d = pir.h2d_stream(w.shape, w.dtype)
        x = ops.host_load(x_h2d)
        y = w * x
        y_d2h = pir.d2h_stream(y.shape, y.dtype)
        ops.host_store(y_d2h, y)
        ops.var_updates.accumulate_(w, pir.constant(c_input,
                                                    pir.dtypes.float32), 1.)

    session, inputs, outputs = run_session(ir, [x], [x_h2d], 1)

    w_data = session.get_tensor_data(w)
    assert np.allclose(outputs[y_d2h], inputs[x_h2d] * w_input)
    assert np.allclose(
        w_data,
        np.array(w_input).astype(np.float32) +
        np.array(c_input).astype(np.float32))

    session.device.detach()


class LinearHostLoad(pir.Module):
    def __init__(self):
        self.W: Tensor = None
        self.b: Tensor = None

        self.h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        self.d2h = pir.d2h_stream((2, 2), pir.float32, name="y_stream")

    def build(self, out_features: int,
              bias: bool = True) -> Tuple[Tensor, ...]:
        x = ops.host_load(self.h2d, "x")
        self.W = graph_input((x.shape[-1], out_features), pir.float32, "W")
        y = x @ self.W
        if bias:
            self.b = graph_input((out_features, ), pir.float32, "b")
            y = y + self.b
        ops.host_store(self.d2h, y)
        return self.W, self.b


def test_session_multi_iteration():
    ir = pir.Ir()
    bps = 8
    with ir.main_graph:
        W_data = np.random.normal(0, 0.1, (2, 2)).astype(np.float32)
        W = pir.variable(W_data, name="W")
        b_data = np.random.normal(0, 0.4, (2)).astype(np.float32)
        b = pir.variable(b_data, name="b")

        linear = LinearHostLoad()
        linear_graph = ir.create_graph(linear, out_features=2)

        W, b = ops.repeat(linear_graph,
                          bps,
                          inputs_dict={
                              linear.W: W,
                              linear.b: b
                          })

    session, inputs, outputs = run_session(ir, [linear.h2d.spec], [linear.h2d],
                                           bps)

    input_: np.ndarray = inputs[linear.h2d]
    output: np.ndarray = outputs[linear.d2h]
    for i in range(bps):
        assert np.allclose(output[i, ...], (input_[i, ...] @ W_data) + b_data)

    session.device.detach()


def run_session(ir, input_tensors, input_d2hs, num_host_transfers):
    ir.num_host_transfers = num_host_transfers

    session = pir.Session(ir, device_desc="ipu_model")

    inputs = {}
    for t, t_d2h in zip(input_tensors, input_d2hs):
        shape = (num_host_transfers,
                 ) + t.shape if num_host_transfers > 1 else t.shape
        t_input = np.random.normal(0, 0.4, shape).astype(t.dtype.as_numpy())
        inputs[t_d2h] = t_input

    outputs = session.run(inputs)
    # Could also do `outputs = session.create_host_outputs()`
    # session.run_with_outputs(inputs, outputs)
    return session, inputs, outputs
