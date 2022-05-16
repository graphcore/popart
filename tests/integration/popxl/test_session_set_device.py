# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from itertools import product, chain
import pytest
import popxl
from popxl import ops
from popxl.utils import _to_device_info
from popxl_test_device_helpers import get_test_device_with_timeout


def create_session(device_type: str, ipus: int = 1):
    ir = popxl.Ir()
    ir.replication_factor = ipus
    with ir.main_graph:
        h2d = popxl.h2d_stream((), popxl.float32, "image")
        x = ops.host_load(h2d, "image")
        d2h = popxl.d2h_stream((), popxl.float32)
        ops.host_store(d2h, x + 1)
    return popxl.Session(ir, device_type)


@pytest.mark.parametrize(
    "test_case",
    (("ipu_hw", "ipu_hw"), ("ipu_model", "ipu_model"), ("ipu_hw", "ipu_model"),
     ("ipu_hw", "cpu"), ("ipu_model", "ipu_hw"), ("cpu", "ipu_hw")))
def test_check_compatibility(test_case):
    # Note poplar allows for `ipu_model -> cpu`
    compile_type, run_type = test_case
    session = create_session(compile_type)

    if run_type == "ipu_hw":
        device = get_test_device_with_timeout(numIpus=1)
    else:
        device = _to_device_info("ipu_model")

    session._set_device(device)

    if compile_type != run_type:
        with pytest.raises(Exception):
            with session:
                pass
    else:
        with session:
            pass


def test_check_different_ipu_incompatibility():
    session = create_session("ipu_model", 2)
    device = _to_device_info("ipu_model", 1)
    session._set_device(device)
    with pytest.raises(Exception):
        with session:
            pass
