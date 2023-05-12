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
from popxl.utils import host_pow2scale_cast_to_fp8
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
        lhs = host_pow2scale_cast_to_fp8(lhs, dtype, 0, True)
        rhs = host_pow2scale_cast_to_fp8(rhs, dtype, 0, True)

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
            input_val = host_pow2scale_cast_to_fp8(input_val, dtype, 0, True)

        np.testing.assert_equal(input_array, input_val)
        # There will be slight loss of precision for float8 types as the inputs
        # were downcasted to float8 on device.
        np.testing.assert_allclose(
            tensor_dict[copied_output], output_val, 0.05, 0.05, equal_nan=True
        )


def loaded_saved_executable(capfd: pytest.CaptureFixture) -> Tuple[bool, str]:
    """
    Check whether an executable was loaded or not.

    The output log of the POPART log level DEBUG will be used to check this.

    Args:
        capfd (pytest.CaptureFixture): The output captured from the file descriptors

    Returns:
        bool, str: True if the executable was loaded, False otherwise
            Also returns the standard error of the test.
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
    return (not started_engine_compilation), stderr


def inferred_hash_was_loaded(stderr: str) -> bool:
    """
    Check whether an executable with an inferred hash was successfully loaded

    The output log of the POPART log level DEBUG will be used to check this.

    Args:
        stderr (str): The stderr generated by the output captured from the
            file descriptors

    Returns:
        bool: True if the executable was loaded, False otherwise
    """

    loaded_executables = []
    inferred_executable = []
    for line in stderr.splitlines():
        if "Loading serialized PopART executable" in line:
            loaded_executables.append(line.split(" ")[-1])
        elif "PopEF file inferred to have hash" in line:
            inferred_executable.append(line.split(" ")[-1])

    # Assert that we didn't both start a compilation AND load an executable
    assert (
        len(loaded_executables) == 1
    ), "All hashes should be inferred and a single executable loaded"
    return loaded_executables[0] in inferred_executable


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

    is_executable_loaded, _ = loaded_saved_executable(capfd=capfd)
    assert not is_executable_loaded
    # Check that the cache has been saved
    assert len(list(cache_path.glob("**/*.popef"))) == 1

    # Second run. Executable cache used
    ir, input_tensor, copied_output = create_ir_with_copy_var_update(
        lhs, rhs, init_val, dtype=dtype, log2scale=log2scale
    )
    sess = popxl.Session(ir, "ipu_hw")
    run_and_compare(sess, input_tensor, copied_output, lhs, output_val, dtype)

    is_executable_loaded, stderr = loaded_saved_executable(capfd=capfd)
    assert is_executable_loaded
    loaded_executable_was_inferred = inferred_hash_was_loaded(stderr)
    assert loaded_executable_was_inferred


@tu.requires_ipu
def test_bad_cache_files_are_not_loaded(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capfd: pytest.CaptureFixture,
) -> None:
    """Test that bad cache files are not loaded even if they appear to be hash matches based
    on the cache file names.

    Args:
        tmp_path (Path): Temporary directory
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
    """

    def run(size: int):
        """Run a very simple model for a given square matrix multiplication size."""
        lhs = np.random.random([size, size]).astype(np.float32)
        rhs = np.random.random([size, size]).astype(np.float32)

        output_val = np.matmul(lhs, rhs)

        init_val = np.zeros([size, size]).astype(np.float32)

        # First run. Compile and run without executable cache
        ir, input_tensor, copied_output = create_ir_with_copy_var_update(
            lhs, rhs, init_val, dtype=popxl.float32, log2scale=0
        )

        sess = popxl.Session(ir, "ipu_hw")
        run_and_compare(
            sess, input_tensor, copied_output, lhs, output_val, popxl.float32
        )

    cache_path = tmp_path / "saved_graph"
    # Enable model caching
    monkeypatch.setenv("POPXL_CACHE_DIR", str(cache_path))

    np.random.seed(0)
    # Check that no cache lingers from previous tests
    assert len(list(cache_path.glob("**/*.popef"))) == 0
    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel("DEBUG")

    run(size=2)
    is_executable_loaded, _ = loaded_saved_executable(capfd=capfd)
    assert not is_executable_loaded
    # Check that the cache has been saved
    assert len(list(cache_path.glob("**/*.popef"))) == 1
    exec_cache_1 = [*cache_path.glob("**/*.popef")][0]

    # Second run. No Executable cache used
    run(size=3)
    is_executable_loaded, _ = loaded_saved_executable(capfd=capfd)
    assert not is_executable_loaded
    assert len(list(cache_path.glob("**/*.popef"))) == 2
    exec_cache_2 = [c for c in cache_path.glob("**/*.popef") if c != exec_cache_1][0]

    # Generate some bad cache files
    (cache_path / "some-bad-cache-file.popef").write_text("This is not cache")
    # Overwrite the cache file we are going to load with the executable cache from the second run
    exec_cache_1.write_bytes(exec_cache_2.read_bytes())
    # We expect this run to infer the hash -> check the executable cache -> then compile
    run(size=2)
    is_executable_loaded, stderr = loaded_saved_executable(capfd=capfd)
    assert not is_executable_loaded
    assert "Cache file hash did not match the IR hash" in stderr
    assert "Ignoring cache file because it does not contain" in stderr


@tu.requires_ipu
def test_popart_preload(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capfd: pytest.CaptureFixture,
) -> None:
    """
    Test that the cache is preloaded when the appropriate environment variable is set

    Args:
        tmp_path (Path): Temporary directory
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
    """

    def run():
        """Run a very simple model for a given square matrix multiplication size."""
        size = 2
        lhs = np.random.random([size, size]).astype(np.float32)
        rhs = np.random.random([size, size]).astype(np.float32)

        output_val = np.matmul(lhs, rhs)

        init_val = np.zeros([size, size]).astype(np.float32)

        # First run. Compile and run without executable cache
        ir, input_tensor, copied_output = create_ir_with_copy_var_update(
            lhs, rhs, init_val, dtype=popxl.float32, log2scale=0
        )

        sess = popxl.Session(ir, "ipu_hw")
        run_and_compare(
            sess, input_tensor, copied_output, lhs, output_val, popxl.float32
        )

    cache_path = tmp_path / "preload"
    # Enable model caching
    monkeypatch.setenv("POPXL_CACHE_DIR", str(cache_path))

    np.random.seed(0)
    # Check that no cache lingers from previous tests
    assert len(list(cache_path.glob("**/*.popef"))) == 0
    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel("DEBUG")

    # first run, generate the cached executable
    run()
    is_executable_loaded, _ = loaded_saved_executable(capfd=capfd)
    assert not is_executable_loaded
    # Check that the cache has been saved
    assert len(list(cache_path.glob("**/*.popef"))) == 1

    # second run, enable preload
    monkeypatch.setenv("POPART_PRELOAD_POPEF", "full-preload")
    run()
    # check that the preload took place
    _, stderr = capfd.readouterr()
    assert "Completed preload of popef file" in stderr

    # third run, disable preload
    monkeypatch.delenv("POPART_PRELOAD_POPEF", raising=False)
    run()
    # check that preload did not take place
    _, stderr = capfd.readouterr()
    assert "Performing preload of popef file" not in stderr
