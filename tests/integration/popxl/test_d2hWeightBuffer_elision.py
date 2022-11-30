# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from popxl_test_device_helpers import mk_session_with_test_device

import pytest
import numpy as np
import popxl
import popxl.ops as ops
from popxl.dtypes import dtype


def test_d2hWeightBuffer_elision(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    """
    Test that Devicex never creates d2hWeightBuffers for our variables (and
    instead reuses the TensorData buffers).
    Test this is true for on-chip or remote variables, as well as replica grouped
    variables.

    This behaviour is possible because `popxl.Ir` always sets the `enableVariablesCaching` session option
    to false, which enables this optimisation.

    We test this by inspecting the log.
    """

    monkeypatch.setenv("POPART_LOG_LEVEL", "TRACE")

    ir = popxl.Ir(replication=2)

    # Two groups of (2, 2)
    w_h_grouped = np.ones(
        (2, 2, 2),
        dtype=np.float32,
    )
    w_dtype = dtype.as_dtype(w_h_grouped.dtype)

    with ir.main_graph:
        # w_onchip just one group
        w_onchip = popxl.variable(w_h_grouped[0], w_dtype, name="w_onchip")

        # w_remote two groups
        w_rb = popxl.remote_buffer(w_h_grouped[0].shape, w_dtype)
        w_remote = popxl.remote_variable(
            w_h_grouped,
            w_rb,
            replica_grouping=ir.replica_grouping(1, 2),
            name="w_remote",
        )

        w_remote_loaded = ops.remote_load(w_rb, 0)
        w_updated = ops.add_(w_onchip, w_remote_loaded)
        ops.remote_store(w_rb, 0, w_updated)

    # Flush captured log so far to clear it.
    capfd.readouterr()
    # The log message we are looking for gets emitted during
    # `Devicex::loadEngineAndConnectStreams` which will happen when the
    # Session contextmanager enters and does weights_from_host.
    with mk_session_with_test_device(ir) as sess:
        sess.run()
    log = capfd.readouterr().err

    def assert_d2hWeightBuffer_elided(t):
        import re

        # Comes from Devicex::initD2hWeightBuffer
        elision_log_msg = f"Reusing TensorData for d2hWeightBuffer of tensor {t.id}"

        pattern = re.compile(elision_log_msg)
        matches = re.findall(pattern, log)

        assert (
            len(matches) == 1
        ), f"Failed to find d2hWeightBuffer elision message in trace log for tensor {t.id}. Log was:\n{log}"

    assert_d2hWeightBuffer_elided(w_onchip)
    assert_d2hWeightBuffer_elided(w_remote)

    _ = sess.get_tensors_data([w_onchip, w_remote])
