# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest

import popart
import test_util as tu


def test_passing_non_contiguous_to_py_weights_io_should_raise_exception():
    """Reproducer for ~T68462~"""

    W_init = np.arange(6, dtype=np.float32).reshape(2, 3).transpose(1, 0)
    assert not W_init.flags.c_contiguous, "Test assumes W_init would be non-contiguous"

    # Model
    builder = popart.Builder()
    W = builder.addInitializedInputTensor(W_init, "W")
    loss = builder.aiOnnx.reducesumsquare([W])  # dummy, unused

    # Session
    with tu.create_test_device(numIpus=1) as device:
        options = popart.SessionOptions()
        options.constantWeights = False
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            deviceInfo=device,
            dataFlow=popart.DataFlow(1, {loss: popart.AnchorReturnType("ALL")}),
            userOptions=options,
        )
        session.prepareDevice()
        session.weightsFromHost()
        session.weightsToHost()

        # Read back initial weights, deliberately making W_out non-contiguous
        W_out = np.zeros_like(W_init)
        assert (
            not W_out.flags.c_contiguous
        ), "Test assumes W_out would be non-contiguous"

        # Possible solutions for customer (enable one):
        # W_out = np.zeros_like(W_init, order="C")
        # W_out = np.zeros(np.shape(W_init), dtype=W_init.dtype)
        # W_out = np.ascontiguousarray(W_out)

        with pytest.raises(popart.popart_exception) as e_info:
            session.readWeights(popart.PyWeightsIO({W: W_out}))

        assert "it is not c-contiguous" in str(e_info.value)

        # If framework is ever improved to cope rather than error, then original
        # reproducer checked arrays matched:
        # np.testing.assert_equal(W_out, W_init)
