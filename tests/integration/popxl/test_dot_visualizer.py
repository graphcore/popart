# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Test dot visualizer functionality in popxl"""
from typing import Optional
import pytest
from pytest import MonkeyPatch
import os
import tempfile
import numpy as np
import popxl
import popxl.ops as ops
from pathlib import Path

_TENSOR_SHAPE = (3, 11, 5)


@pytest.mark.parametrize("check_name, expected_dot_file_count",
                         (("", 0), ("FINAL", 1), ("FOO:BAR", 2), ("ALL", 3)))
@pytest.mark.parametrize("use_environ", (True, False))
def test_dot_check_with_environ_and_opts(
        monkeypatch: MonkeyPatch, check_name: str,
        expected_dot_file_count: int, use_environ: bool) -> None:
    """Test that the popxl DotVisualizer works with both POPART_DOT_CHECKS and dotChecks.

    Args:
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        check_name (str): The name of the check
        expected_dot_file_count (int): The expected number of dotfiles the input will produce
        use_environ (bool): Uses os.environ if true to set the checks, else it uses session.options
    """
    if use_environ:
        # Set the environment variables
        monkeypatch.setenv("POPART_DOT_CHECKS", check_name)
    else:
        monkeypatch.setenv("POPART_DOT_CHECKS", "")

    # NOTE: Any session options need to be set prior to creating the model
    ir = popxl.Ir()
    ir_pb = ir._pb_ir

    opts = ir_pb.getSessionOptions()
    if not use_environ:
        opts.dotChecks = {*check_name.split(':')}

    run_test(ir=ir,
             save_dir=None,
             expected_dot_file_count=expected_dot_file_count)


def test_automatic_dot_check() -> None:
    """Test the automatic setup of dot_checkpoint.

    Specifically:
    - Test that dot_checkpoint will set dotChecks to ALL
      if neither POPART_DOT_CHECKS nor dotChecks is specified
    - Test that if save_dir if set, then logDir will be set and
      the dot files will be stored in save_dir
    """
    ir = popxl.Ir()

    # Make sure that there are no remnants from other test
    if "POPART_DOT_CHECKS" in os.environ:
        os.environ.pop("POPART_DOT_CHECKS")
    ir._pb_ir.getSessionOptions().dotChecks = set()

    # Test that dot_checkpoint sets dotChecks to ALL
    run_test(ir, save_dir=None, expected_dot_file_count=3)

    # Check that the files will be stored in save_dir
    run_test(ir, save_dir="FooBar", expected_dot_file_count=3)


def run_test(ir: popxl.Ir, save_dir: Optional[str],
             expected_dot_file_count: int) -> None:
    """Run inference and check expected dot file count.

    Args:
        ir (popxl.Ir): The ir to write to
        save_dir (Optional[Path]): Directory to store the dot files in.
        expected_dot_file_count (int): The expected number of dot files
    """
    ir_pb = ir._pb_ir
    opts = ir_pb.getSessionOptions()
    opts.useHostCopyOps = True
    with tempfile.TemporaryDirectory() as tmp_dir:
        check_dir = Path(tmp_dir)

        if save_dir is None:
            # If no save_dir is specified, we just use the tmp_dir
            opts.logDir = str(check_dir)
        else:
            # If a save_dir is specified, dot_checkpoint will specify opts.logDir for us
            check_dir = check_dir.joinpath(save_dir)

        # Obtain the model
        build_model_with_dot_checkpoints(ir, str(check_dir))

        _ = popxl.Session(ir, "ipu_model")

        dot_files = list(check_dir.glob('*.dot'))
        assert len(dot_files) == expected_dot_file_count


def build_model_with_dot_checkpoints(ir: popxl.Ir,
                                     save_dir: Optional[str] = None) -> None:
    """Make a model with 2 dot_checkpoints.

    Args:
        ir (popxl.Ir): The ir to write to
        save_dir (Optional[str], optional): Directory to store the dot files in.
          Defaults to None.

    Returns:
    (tuple): tuple containing:

        ir._pb_ir (_ir.Ir): The underlying IR
        a_h2d (HostToDeviceStream): The host to device stream
        f_d2h (DeviceToHostStream): The device to host stream
    """
    main = ir.main_graph

    with main:
        a_h2d = popxl.h2d_stream(_TENSOR_SHAPE, popxl.float32, name="a_stream")
        a = ops.host_load(a_h2d, "a")

        b = popxl.variable(np.random.rand(*_TENSOR_SHAPE).astype(np.float32),
                           name="b")
        c = ops.add(a, b)
        ir.dot_checkpoint("Foo", save_dir)

        d = popxl.variable(np.random.rand(*_TENSOR_SHAPE).astype(np.float32),
                           name="d")
        e = ops.mul(c, d)
        ir.dot_checkpoint("Bar", save_dir)

        f = ops.gelu(e)

        f_d2h = popxl.d2h_stream(_TENSOR_SHAPE, popxl.float32, name="f_stream")
        ops.host_store(f_d2h, f)
