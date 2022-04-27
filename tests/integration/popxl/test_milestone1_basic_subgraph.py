# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Milestone 1: Basic subgraph
"""

import popxl
from popxl.streams import HostToDeviceStream, DeviceToHostStream
import popxl.ops as ops
import numpy as np
from typing import Tuple


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


_INPUT_SHAPE = (2, 16)


# Build model using popxl API, then return the underlying Ir.
# Also returns the streams for the input and output tensors, and the data of the
# variables.
def build_model(
) -> Tuple[popxl.Ir, HostToDeviceStream, DeviceToHostStream, np.array, np.
           array]:
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        x_h2d = popxl.h2d_stream(_INPUT_SHAPE, popxl.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        out_features = x.shape[-1]
        weight_shape = (out_features, out_features)
        bias_shape = (out_features, )
        W_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        b_data = np.zeros(bias_shape, dtype=np.float32)

        W = popxl.variable(W_data, name="W")
        b = popxl.variable(b_data, name="b")

        lin = Linear()
        lin_graph = ir.create_graph(lin, x, out_features=out_features)

        y, = ops.call(lin_graph, x, inputs_dict={lin.W: W, lin.b: b})

        y_d2h = popxl.d2h_stream(_INPUT_SHAPE, popxl.float32, name="y_stream")
        ops.host_store(y_d2h, y)

    return ir, x_h2d, y_d2h, W_data, b_data


def test_basic_subgraph():
    """
    Builds and executes the following (pseudocode):
    x = ones(shape=(2, 16))
    W = random_normal(shape=(16, 16))
    b = zeros(shape=(16,))
    y = x @ W + b

    Compute in numpy: y_expected = x.data() @ W.data() + b.data()
    assert y == y_expected
    """
    ir, x_h2d, y_d2h, W_data, b_data = build_model()

    ir, x_h2d, y_d2h, W_data, b_data = build_model()

    ir.num_host_transfers = 1

    session = popxl.Session(ir, 'ipu_model')
    print(f'expected_inputs = {session.expected_inputs()}')

    # Create data for input x
    x_data = np.ones(_INPUT_SHAPE, dtype=np.float32)

    inputs = {x_h2d: x_data}

    with session:
        outputs = session.run(inputs)

    expected_y = np.matmul(x_data, W_data) + b_data
    y = outputs[y_d2h]

    print(f'y = {outputs[y_d2h]}')

    assert y.shape == expected_y.shape
    assert y.dtype == expected_y.dtype
    assert np.allclose(y, expected_y)
