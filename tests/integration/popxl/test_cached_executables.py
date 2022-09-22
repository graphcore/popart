# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
Module containing tests and helper functions related to saving executables for PopXL.

See also tests/integration/saved_executable.py for PopART tests
"""

from typing import Tuple
from pytest import MonkeyPatch
import pytest
import popxl
import sys
import numpy as np
import popart
from pathlib import Path

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import test_util as tu


def create_ir_with_copy_var_update(
    input_val: float, add_val: int, init_val: float
) -> Tuple[popxl.Ir, popxl.Tensor, popxl.Tensor]:
    """Creates an IR where the output of an operation is copied to another tensor.

    Args:
        input_val (float): Value to set in the input tensor
        add_val (int): Value to add to the input tensor
        init_val (float): Value to set copied_output tensor

    Returns:
        Tuple[popxl.Ir, popxl.Tensor, popxl.Tensor]: Tuple containing:
            - The created IR
            - The created input tensor
            - The copied output tensor
    """
    ir = popxl.Ir()
    with ir.main_graph:
        input_tensor = popxl.variable(input_val, name="input_tensor")
        output_tensor = input_tensor + add_val

        copied_output = popxl.variable(
            init_val, dtype=popxl.float32, name="copied_output"
        )
        popxl.ops.var_updates.copy_var_update_(copied_output, output_tensor)
    return ir, input_tensor, copied_output


def run_and_compare(
    sess: popxl.Session,
    input_tensor: popxl.Tensor,
    copied_output: popxl.Tensor,
    input_val: float,
    output_val: float,
) -> None:
    """Run the session and compare it with the expected results.

    Args:
        sess (popxl.Session): The session to run
        input_tensor (popxl.Tensor): The input tensor of the session
        copied_output (popxl.Tensor): The copied output tensor of the session
        input_val (float): The value the input tensor is set with
        output_val (float): The expected output value
    """
    with sess:
        sess.run()
        variables = sess.ir.main_graph.variables
        tensor_dict = sess.get_tensors_data(variables)

        assert np.isclose(tensor_dict[input_tensor], input_val)
        assert np.isclose(tensor_dict[copied_output], output_val)


def loaded_saved_executable(capfd: pytest.CaptureFixture) -> bool:
    """
    Check whether an executable was loaded or not.

    The output log of the POPART log level DEBUG will be used to check this.

    Args:
        capfd (pytest.CaptureFixture): The output captured from the file descriptors

    Returns:
        bool: True if the executable was loaded, False otherwise
    """
    _, stderr = capfd.readouterr()
    started_engine_compilation = False
    loaded_poplar_executable = False
    for line in stderr.splitlines():
        if "Starting compilation" in line:
            started_engine_compilation = True
        elif "Loading serialized PopART executable" in line:
            loaded_poplar_executable = True

    # Assert that we didn't both start a compilation AND load an executable
    assert started_engine_compilation != loaded_poplar_executable
    return not started_engine_compilation


@tu.requires_ipu
def test_get_tensors_data(
    tmp_path: Path, monkeypatch: MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    """Test that get_tensors_data is working with engine caching.

    Args:
        tmp_path (Path): Temporary directory
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
    """
    cache_path = tmp_path / "saved_graph"
    # Enable model caching
    monkeypatch.setenv("POPXL_CACHE_DIR", str(cache_path))

    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel("DEBUG")

    input_val = 5.0
    add_val = 10
    output_val = input_val + add_val
    init_val = 0.0

    # Check that no cache lingers from previous tests
    assert len(list(cache_path.glob("**/*.popef"))) == 0

    # First run. Compile and run without executable cache
    ir, input_tensor, copied_output = create_ir_with_copy_var_update(
        input_val, add_val, init_val
    )
    sess = popxl.Session(ir, "ipu_hw")
    run_and_compare(sess, input_tensor, copied_output, input_val, output_val)

    assert not loaded_saved_executable(capfd=capfd)
    # Check that the cache has been saved
    assert len(list(cache_path.glob("**/*.popef"))) == 1

    # Second run. Executable cache used
    ir, input_tensor, copied_output = create_ir_with_copy_var_update(
        input_val, add_val, init_val
    )
    sess = popxl.Session(ir, "ipu_hw")
    # This was failing before ~T62680~. copied_output was not input_val + add_val
    # but init_val.
    run_and_compare(sess, input_tensor, copied_output, input_val, output_val)

    assert loaded_saved_executable(capfd=capfd)
