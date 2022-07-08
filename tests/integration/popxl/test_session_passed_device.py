# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import re
import pytest
import popxl
from popxl import ops
from popxl_test_device_helpers import get_test_device_with_timeout


@pytest.fixture(scope="module", autouse=True)
def capture_popart_logging():
    from unittest.mock import patch

    with patch.dict(
        "os.environ",
        {
            "POPART_LOG_LEVEL": "DEBUG",
        },
    ):
        yield


def test_session_passed_device(capfd):
    """Tests that variables are initialised correctly when
    an attached device is passed to the Session constructor."""

    def num_weights_from_host():
        # This is very brittle
        weights_to_host_log_str = "Writing weights from host, "
        pattern = re.compile(weights_to_host_log_str)

        log = capfd.readouterr().err

        # Go to start of file
        matches = re.findall(pattern, log)
        return len(matches)

    capfd.readouterr()

    ir = popxl.Ir()
    with ir.main_graph:
        v = popxl.variable(1, popxl.float32)
        d2h = popxl.d2h_stream(v.shape, v.dtype)
        ops.host_store(d2h, v)

    device = get_test_device_with_timeout(1)
    assert device.tryAttachUntilTimeout()

    with popxl.Session(ir, device) as sess:
        assert sess.run()[d2h] == 1
        assert (
            num_weights_from_host() == 1
        ), "Expected exactly 1 WeightsToHost when constructing Session with an attached Device."

    # Explicitly detach
    device.detach()
