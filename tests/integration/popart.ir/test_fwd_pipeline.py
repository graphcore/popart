# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Testing of pipelining with one forward pass for a simple model without pipeline stages."""

from typing import Tuple
import torch
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from popart.ir.streams import HostToDeviceStream, DeviceToHostStream
import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def reference(t_data: np.array, weights_data: np.array) -> np.array:
    """Return result of equivalent calculation of the test in pytorch.

    Args:
        t_data (np.array): Input tensor data
        weights_data (np.array): Input tensor weights

    Returns:
        np.array: The result of the pytorch operations
    """
    t_in = torch.from_numpy(t_data)
    weights = torch.from_numpy(weights_data)

    t_1 = torch.matmul(t_in, weights)
    t_2 = torch.nn.functional.gelu(t_1)
    t_out = torch.nn.functional.softmax(t_2, dim=1)

    return t_out.numpy()


def build_model(weights_data: np.array, input_shape: Tuple[int]
                ) -> Tuple[_ir.Ir, HostToDeviceStream, DeviceToHostStream]:
    """Build the model using popart.ir API.
    
    Args:
        weights_data (np.array): The (non-streamed) data of the weights
        input_shape (tuple): The shape of the streamed input tensor

    Returns:
    (tuple): tuple containing:

        ir._pb_ir(_ir.Ir): The underlying IR
        t_in_h2d(HostToDeviceStream): The input stream of t_in
        t_out_d2h (DeviceToHostStream): The output stream of t_out
    """
    ir = pir.Ir()

    main = ir.main_graph()
    with main:
        weights = pir.variable(weights_data, name="weights")
        # Load t_in from host
        t_in_h2d = pir.h2d_stream(input_shape, pir.float32, name="t_in_stream")

        # Operations on IPU 0
        with pir.virtual_graph(0):
            t_in = ops.host_load(t_in_h2d, "t_in")
            t_1 = ops.matmul(t_in, weights)
            # Copy to IPU 1
            t_1_c = ops.ipu_copy(t_1, 1)

        # Operations on IPU 1
        with pir.virtual_graph(1):
            t_2 = ops.gelu(t_1_c)
            # Copy to IPU 2
            t_2_c = ops.ipu_copy(t_2, 2)

        # Operations on IPU 2
        with pir.virtual_graph(2):
            t_out = ops.softmax(t_2_c, axis=1)
            t_out_d2h = pir.d2h_stream(t_out.shape,
                                       pir.float32,
                                       name="t_out_stream")
            ops.host_store(t_out_d2h, t_out)

    return ir._pb_ir, t_in_h2d, t_out_d2h


def test_fwd_pipeline():
    """
    Test one forward pass of a simple pipeline model in serial.

    The test compares the outcome from popart.ir with outcome from pytorch
    """
    # Create the needed tensor shapes
    input_shape = (2, 16)
    w_shape = (input_shape[-1], 4)
    # Create inputs
    weights_data = np.random.normal(0, 0.1, w_shape).astype(np.float32)
    # This will be streamed to popart.ir
    t_data = np.random.normal(0, 0.1, input_shape).astype(np.float32)

    # Build the model
    ir, t_in_h2d, t_out_d2h = build_model(weights_data, input_shape)

    # Get the tensor ids
    t_in_id = t_in_h2d.tensor_id()
    t_out_id = t_out_d2h.tensor_id()

    # Set the data flow
    bps = 1
    data_flow = popart.DataFlow(bps,
                                {t_out_id: popart.AnchorReturnType("All")})
    ir.setDataFlow(data_flow)

    # Set options
    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    # Prepare graph
    ir.updateVertices()
    ir.setIsPrepared()

    # Create an IR inference session
    session = popart.InferenceSession.fromIr(
        ir=ir, deviceInfo=tu.create_test_device(numIpus=3))

    session.prepareDevice()

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    stepio = popart.PyStepIO({t_in_id: t_data}, anchors)
    session.weightsFromHost()
    session.run(stepio)

    # Compare outcome from popart.ir with outcome from pytorch
    expected_t_out = reference(t_data, weights_data)
    t_out = anchors[t_out_id]
    assert t_out.shape == expected_t_out.shape
    assert t_out.dtype == expected_t_out.dtype
    assert np.allclose(t_out, expected_t_out)
