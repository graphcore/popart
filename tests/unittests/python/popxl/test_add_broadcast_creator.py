# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import re
import pytest
import popxl
from popxl import ops
import numpy as np


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


@pytest.mark.parametrize(
    ("lhs", "rhs", "creator"),
    (
        ((4, 4), (4,), 1),
        ((4, 8), (8,), 1),
        ((4, 8), (4, 1), 1),
        ((4, 4), (4, 4), 0),  # No broadcast
        ((4, 1), (1, 4), 0),  # Both Operands broadcast
    ),
)
def test_add_broadcast_creator(capfd, lhs, rhs, creator):
    capfd.readouterr()

    ir = popxl.Ir()
    with ir.main_graph:
        result = popxl.variable(np.ones(lhs)) + popxl.variable(np.ones(rhs))
        d2h = popxl.d2h_stream(result.shape, result.dtype)
        ops.host_store(d2h, result)
    popxl.Session(ir)

    pattern = re.compile("createBroadcastOperand")
    log = capfd.readouterr().err
    assert len(re.findall(pattern, log)) == creator
