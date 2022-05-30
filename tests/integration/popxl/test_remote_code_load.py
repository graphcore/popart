# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Basic subgraph but with code copy.
"""

from typing import List, Tuple

import numpy as np
import re
import pva

import popxl
import popxl.ops as ops
from popxl.streams import DeviceToHostStream, HostToDeviceStream


class Linear(popxl.Module):
    W: popxl.Tensor
    b: popxl.Tensor

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


def build_model(
        remote_code_load: bool = True
) -> Tuple[popxl.Ir, HostToDeviceStream, DeviceToHostStream, np.ndarray, np.
           ndarray]:
    """Build model using popxl API, then return the underlying Ir. Also returns the streams for the
    input and output tensors, and the data of the variables.

    Args:
        remote_code_load (bool, optional): Wehter to include a remote_code_load op in the model.
            Defaults to True.

    Returns:
        Tuple[popxl.Ir, HostToDeviceStream, DeviceToHostStream, np.ndarray, np. ndarray]:
            A tuple of the IR, h2d stream, d2h stream, and weights and bias data.
    """
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

        with popxl.in_sequence():
            if remote_code_load:
                ops.remote_code_load(lin_graph, destination="executable")
            y, = ops.call(lin_graph, x, inputs_dict={lin.W: W, lin.b: b})

        y_d2h = popxl.d2h_stream(_INPUT_SHAPE, popxl.float32, name="y_stream")
        ops.host_store(y_d2h, y)

    return ir, x_h2d, y_d2h, W_data, b_data


def basic_subgraph(
        remote_code_load: bool = True) -> List[pva.pva_core.ExecutionStep]:
    """Builds and executes the following (pseudocode):
    x = ones(shape=(2, 16))
    W = random_normal(shape=(16, 16))
    b = zeros(shape=(16,))
    y = x @ W + b

    Compute in numpy: y_expected = x.data() @ W.data() + b.data()
    assert y == y_expected

    Args:
        remote_code_load (bool): Whether to use remote_code_load ops in the run. Defaults to True.

    Returns:
        List[pva.pva_core.ExecutionStep]: A list of the execution steps from libpva. See the libpva
    documentation for details of how these can be read.
    """

    ir, x_h2d, y_d2h, W_data, b_data = build_model(
        remote_code_load=remote_code_load)

    ir.num_host_transfers = 1

    # Create data for input x
    x_data = np.ones(_INPUT_SHAPE, dtype=np.float32)

    inputs = {x_h2d: x_data}

    session = popxl.Session(ir, 'ipu_model')
    print(f'expected_inputs = {session.expected_inputs()}')

    with session:
        outputs = session.run(inputs)

    expected_y = np.matmul(x_data, W_data) + b_data
    y = outputs[y_d2h]

    print(f'y = {outputs[y_d2h]}')

    assert y.shape == expected_y.shape
    assert y.dtype == expected_y.dtype
    assert np.allclose(y, expected_y)

    report = session._pb_session.getReport()

    # Get all the streams. Just StreamCopyMid is fine as they with be in groups of {Begin, Mid, End}
    streams = [
        s for s in report.execution.steps
        if s.program.type == pva.pva_core.Program.Type.StreamCopyMid
    ]

    return streams


def test_remote_code_load() -> None:
    with_remote_code_load = basic_subgraph(True)
    without_remote_code_load = basic_subgraph(False)

    # There should be 1 more stream copy in the model with the code copy.
    assert len(with_remote_code_load) == len(without_remote_code_load) + 1

    # One of the debug names must match this string. Account for changes in scheduling and
    # name changes in future, so match one in any position and use a regex.
    match_str = r"remote_code_load\/[0-9]{3}\/(.*) Remote -> Device , graph: (.*)\(0\)"
    assert any(
        [re.match(match_str, s.program.name) for s in with_remote_code_load])
