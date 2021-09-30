# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Milestone 1: Basic subgraph
"""

import popart.ir as pir
import popart.ir.ops as ops

import popart._internal.ir as _ir

import popart

import numpy as np

from typing import Tuple

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


class ScaleNShift(pir.Module):
    def __init__(self):
        self.W: pir.Tensor = None
        self.b: pir.Tensor = None

    def build(self, x: pir.Tensor, out_features: int,
              bias: bool = True) -> pir.Tensor:
        self.W = pir.subgraph_input(pir.float32, (x.shape[-1], out_features),
                                    "W")
        y = ops.mul(x, self.W)  # TODO: Change to matmul once implemented
        if bias:
            self.b = pir.subgraph_input(pir.float32, (out_features, ), "b")
            y = y + self.b
        return y


_INPUT_SHAPE = (16, 16)


# Build model using popart.ir API, then return the underlying Ir.
# Also returns the streams for the input and output tensors, and the data of the
# variables.
def build_model() -> Tuple[_ir.Ir, pir.HostToDeviceStream, pir.
                           DeviceToHostStream, np.array, np.array]:
    ir = pir.Ir()

    main = ir.main_graph()
    with main:
        # NOTE: Change to (2, 16) for matmul
        x_h2d = pir.h2d_stream(pir.float32, _INPUT_SHAPE, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        W_data = np.random.normal(0, 0.1, _INPUT_SHAPE).astype(np.float32)
        b_data = np.zeros(16, dtype=np.float32)

        W = pir.variable(W_data, name="W")
        b = pir.variable(b_data, name="b")

        ss = ScaleNShift()
        ss_graph = ir.get_graph(ss, x, out_features=16)

        y = ops.call(ss_graph, x, subgraph_in_to_parent_in={ss.W: W, ss.b: b})

        y_d2h = pir.d2h_stream(pir.float32, _INPUT_SHAPE, name="x_stream")
        ops.host_store(y_d2h, y)

    return ir._pb_ir, x_h2d, y_d2h, W_data, b_data


def test_basic_subgraph():
    """
    Builds and executes the following (pseudocode):
    x = ones(shape=(16, 16))
    W = random_normal(shape=(16, 16))
    b = zeros(shape=(16,))
    y = x * W + b

    Compute in numpy: y_expected = x.data() * W.data() + b.data()
    assert y == y_expected
    """
    ir, x_h2d, y_d2h, W_data, b_data = build_model()

    x_id = x_h2d.tensor_id()
    y_id = y_d2h.tensor_id()

    bps = 1
    dataFlow = popart.DataFlow(bps, {y_id: popart.AnchorReturnType("All")})
    ir.setDataFlow(dataFlow)

    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    ir.updateVertices()
    ir.setIsPrepared()

    session = popart.InferenceSession.fromIr(
        ir=ir, deviceInfo=tu.create_test_device())

    session.prepareDevice()

    # Create data for input x
    x_data = np.ones(_INPUT_SHAPE, dtype=np.float32)

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    stepio = popart.PyStepIO({x_id: x_data}, anchors)

    session.weightsFromHost()
    session.run(stepio)

    expected_y = np.multiply(x_data, W_data) + b_data
    y = anchors[y_id]

    assert y.shape == expected_y.shape
    assert y.dtype == expected_y.dtype
    assert np.allclose(y, expected_y)
