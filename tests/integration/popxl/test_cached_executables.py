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
import popxl.ops as ops
from popxl.utils import host_pow2scale_then_cast
from utils import get_representable_float_8_np_array

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import test_util as tu


def create_ir_with_copy_var_update(
    lhs: np.ndarray,
    rhs: np.ndarray,
    init_val: np.ndarray,
    dtype: popxl.dtype,
    log2scale: int,
) -> Tuple[popxl.Ir, popxl.Tensor, popxl.Tensor]:
    """Creates an IR where the output of an operation is copied to another tensor.

    Args:
        lhs (np.ndarray): Value to set in the input tensor
        rhs (np.ndarray): Value to matmul with the input tensor
        init_val (np.ndarray): Value to set copied_output tensor
        dtype: (popxl.dtype): Test for float32 and special cases float8_143, float8_152
        log2scale (int): log scaling to use. Only used for float8 types.

    Returns:
        Tuple[popxl.Ir, popxl.Tensor, popxl.Tensor]: Tuple containing:
            - The created IR
            - The created input tensor
            - The copied output tensor
    """
    ir = popxl.Ir()
    if dtype in [popxl.float8_143, popxl.float8_152]:
        # Create the float8 data on host before moving it to device.
        # Note here we do not scale during the cast, as the only scaling
        # is performed during the matmul.
        lhs = host_pow2scale_then_cast(lhs, dtype, 0, True)
        rhs = host_pow2scale_then_cast(rhs, dtype, 0, True)

    with ir.main_graph:
        lhs_t = popxl.variable(lhs, name="lhs", dtype=dtype)
        rhs_t = popxl.variable(rhs, name="rhs", dtype=dtype)

        if dtype in [popxl.float8_143, popxl.float8_152]:
            log2scale_tensor = popxl.constant(log2scale, popxl.int32)
            output_tensor = ops.matmul_pow2scaled(lhs_t, rhs_t, log2scale_tensor)
        else:
            output_tensor = ops.matmul(lhs_t, rhs_t)

        out_type = popxl.float32 if dtype == popxl.float32 else popxl.float16
        copied_output = popxl.variable(init_val, dtype=out_type, name="copied_output")
        popxl.ops.var_updates.copy_var_update_(copied_output, output_tensor)
    return ir, lhs_t, copied_output


def run_and_compare(
    sess: popxl.Session,
    input_tensor: popxl.Tensor,
    copied_output: popxl.Tensor,
    input_val: np.ndarray,
    output_val: np.ndarray,
    dtype: popxl.dtype,
) -> None:
    """Run the session and compare it with the expected results.

    Args:
        sess (popxl.Session): The session to run
        input_tensor (popxl.Tensor): The input tensor of the session
        copied_output (popxl.Tensor): The copied output tensor of the session
        input_val (float): The value the input tensor is set with
        output_val (float): The expected output value
        dtype: (popxl.dtype): Test for float32 and special cases float8_143, float8_152
    """
    with sess:
        sess.run()
        variables = sess.ir.main_graph.variables
        tensor_dict = sess.get_tensors_data(variables)

        input_array = tensor_dict[input_tensor]

        if dtype in [popxl.float8_143, popxl.float8_152]:
            input_val = host_pow2scale_then_cast(input_val, dtype, 0, True)

        np.testing.assert_equal(input_array, input_val)
        # There will be slight loss of precision for float8 types as the inputs
        # were downcasted to float8 on device.
        np.testing.assert_allclose(
            tensor_dict[copied_output], output_val, 0.05, 0.05, equal_nan=True
        )


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
@pytest.mark.parametrize("dtype", [popxl.float32, popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("log2scale", [-1, 0, 1])
def test_get_tensors_data(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capfd: pytest.CaptureFixture,
    dtype: popxl.dtype,
    log2scale: int,
) -> None:
    """Test that get_tensors_data is working with engine caching.

    Args:
        tmp_path (Path): Temporary directory
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
        dtype (popxl.dtype): Test for float32 and special cases float8_143, float8_152
        log2scale (int): log scaling to use. Only used for float8 types.
    """

    if (dtype == popxl.float32) and (log2scale != 0):
        pytest.skip("Only test float32 once as log2scale is not used.")
    cache_path = tmp_path / "saved_graph"
    # Enable model caching
    monkeypatch.setenv("POPXL_CACHE_DIR", str(cache_path))

    np.random.seed(0)

    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel("DEBUG")

    if dtype == popxl.float32:
        lhs = np.random.random([2, 2]).astype(np.float32)
        rhs = np.random.random([2, 2]).astype(np.float32)

        output_val = np.matmul(lhs, rhs)
    else:
        # Note: No log2scale here, as we scale inside the scaled matmul.
        # Don't allow nans as it will mess up the matmul.
        lhs = get_representable_float_8_np_array([2, 2], dtype, 0, False).astype(
            np.float32
        )
        rhs = get_representable_float_8_np_array([2, 2], dtype, 0, False).astype(
            np.float32
        )
        # Try to emulate the float8 matmul
        output_val = np.matmul(lhs.astype(np.float16), rhs.astype(np.float16)) * pow(
            2, log2scale
        )
        output_val = output_val.astype(np.float16)

    init_val = np.zeros([2, 2]).astype(np.float32)

    # Check that no cache lingers from previous tests
    assert len(list(cache_path.glob("**/*.popef"))) == 0

    # First run. Compile and run without executable cache
    ir, input_tensor, copied_output = create_ir_with_copy_var_update(
        lhs, rhs, init_val, dtype=dtype, log2scale=log2scale
    )

    sess = popxl.Session(ir, "ipu_hw")
    run_and_compare(sess, input_tensor, copied_output, lhs, output_val, dtype)

    assert not loaded_saved_executable(capfd=capfd)
    # Check that the cache has been saved
    assert len(list(cache_path.glob("**/*.popef"))) == 1

    # Second run. Executable cache used
    ir, input_tensor, copied_output = create_ir_with_copy_var_update(
        lhs, rhs, init_val, dtype=dtype, log2scale=log2scale
    )
    sess = popxl.Session(ir, "ipu_hw")
    run_and_compare(sess, input_tensor, copied_output, lhs, output_val, dtype)

    assert loaded_saved_executable(capfd=capfd)
