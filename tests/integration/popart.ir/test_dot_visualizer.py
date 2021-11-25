# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Test dot visualizer functionality in popart.ir"""
import pytest
import popart
import os
import tempfile
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

_TENSOR_SHAPE = (3, 11, 5)


@pytest.mark.parametrize("check_name, expected_dot_file_count",
                         (("", 0), ("FINAL", 1), ("FOO:BAR", 2), ("ALL", 3)))
@pytest.mark.parametrize("use_environ", (True, False))
def test_dot_visualizer_ir(check_name: str, expected_dot_file_count: int,
                           use_environ: bool) -> None:
    """Test the DotVisualizer binding in popart.ir.

    Args:
        check_name (str): The name of the check
        expected_dot_file_count (int): The expected number of dotfiles the input will produce
        use_environ (bool): Uses os.environ if true to set the checks, else it uses session.options
    """
    if use_environ:
        # Set the environment variables
        os.environ['POPART_DOT_CHECKS'] = check_name
    else:
        os.environ['POPART_DOT_CHECKS'] = ''

    # NOTE: Any session options need to be set prior to creating the model
    ir = pir.Ir()
    ir_pb = ir._pb_ir

    opts = ir_pb.getSessionOptions()
    opts.useHostCopyOps = True
    if not use_environ:
        opts.dotChecks = {*check_name.split(':')}

    with tempfile.TemporaryDirectory() as tmp_dir:
        opts.logDir = tmp_dir

        # Obtain the model
        build_model_with_dot_checkpoints(ir)

        # NOTE: This will set the FINAL check
        _ = popart.InferenceSession.fromIr(ir=ir_pb,
                                           deviceInfo=tu.create_test_device())

        dotFiles = list(Path(tmp_dir).glob('*.dot'))
        assert len(dotFiles) == expected_dot_file_count


def build_model_with_dot_checkpoints(ir: pir.Ir) -> None:
    """Make a model with 2 dot_checkpoints.

    Args:
        ir (pir.Ir): The ir to write to

    Returns:
    (tuple): tuple containing:

        ir._pb_ir (_ir.Ir): The underlying IR
        a_h2d (HostToDeviceStream): The host to device stream
        f_d2h (DeviceToHostStream): The device to host stream
    """
    main = ir.main_graph()

    with main:
        a_h2d = pir.h2d_stream(_TENSOR_SHAPE, pir.float32, name="a_stream")
        a = ops.host_load(a_h2d, "a")

        b = pir.variable(np.random.rand(*_TENSOR_SHAPE).astype(np.float32),
                         name="b")
        c = ops.add(a, b)
        ir.dot_checkpoint("Foo")

        d = pir.variable(np.random.rand(*_TENSOR_SHAPE).astype(np.float32),
                         name="d")
        e = ops.mul(c, d)
        ir.dot_checkpoint("Bar")

        f = ops.gelu(e)

        f_d2h = pir.d2h_stream(_TENSOR_SHAPE, pir.float32, name="f_stream")
        ops.host_store(f_d2h, f)
