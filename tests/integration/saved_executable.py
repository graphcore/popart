# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
"""
Module containing tests and helper functions related to saving executables for PopART.

See also tests/integration/popxl/test_cached_executables.py for PopXL tests
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional
from typing_extensions import Literal

import numpy as np
import pytest
from pytest import MonkeyPatch
import popart
import pva

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def get_add_model() -> Tuple[bytes, str, str, str]:
    """Build a model that adds two tensors.

    Returns:
        Tuple: A tuple containing:
            - The model proto
            - The tensor name of the left hand side operand
            - The tensor name of the right hand side operand
            - The tensor name of the output tensor
    """
    # Create a builder and construct a graph
    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT", [3])

    lhs = builder.addInputTensor(data_shape)
    rhs = builder.addInputTensor(data_shape)

    output = builder.aiOnnx.add([lhs, rhs])
    output = builder.aiOnnx.identity([output])

    builder.addOutputTensor(output)

    proto = builder.getModelProto()

    return proto, lhs, rhs, output


def get_subtract_model() -> Tuple[bytes, str, str, str]:
    """Build a model that subtracts two tensors.

    Returns:
        Tuple: A tuple containing:
            - The model proto
            - The tensor name of the left hand side operand
            - The tensor name of the right hand side operand
            - The tensor name of the output tensor
    """
    # Create a builder and construct a graph
    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT", [3])

    lhs = builder.addInputTensor(data_shape)
    rhs = builder.addInputTensor(data_shape)

    output = builder.aiOnnx.sub([lhs, rhs])
    output = builder.aiOnnx.identity([output])

    builder.addOutputTensor(output)

    proto = builder.getModelProto()

    return proto, lhs, rhs, output


def create_inference_session(
        device: popart.DeviceInfo,
        bps: int,
        opts: Optional[popart.SessionOptions] = None,
        model: Literal["add_model", "subtract_model"] = "add_model"
) -> Tuple[popart.InferenceSession, str, str, str]:
    """Create an inference session.

    Args:
        device (popart.DeviceInfo): The device info to be used for the session
        bps (int): Batches per step
        opts (Optional[popart.SessionOptions], optional): The options to be used for the session.
          Defaults to None.
        model (Literal['add_model', 'subtract_model'], optional): Model to use.
          Defaults to "add_model".

    Raises:
        ValueError: If an unsupported model is specified

    Returns:
        Tuple: A tuple containing:
            - The inference session object
            - The tensor name of the left hand side operand
            - The tensor name of the right hand side operand
            - The tensor name of the output tensor
    """
    if model == "add_model":
        proto, lhs, rhs, output = get_add_model()
    elif model == "subtract_model":
        proto, lhs, rhs, output = get_subtract_model()
    else:
        raise ValueError(f"Model '{model}' not supported")

    # Describe how to run the model
    data_flow = popart.DataFlow(bps, {output: popart.AnchorReturnType("All")})

    if opts is None:
        opts = popart.SessionOptions()

    # Create a session to compile and execute the graph
    return popart.InferenceSession(fnModel=proto,
                                   dataFlow=data_flow,
                                   userOptions=opts,
                                   deviceInfo=device), lhs, rhs, output


def run_session_and_check_result(
        session: popart.InferenceSession,
        bps: int,
        lhs: str,
        rhs: str,
        output: str,
        model: Literal["add_model", "subtract_model"] = "add_model"):
    """Run the session and check the result

    Args:
        session (popart.InferenceSession): The session to run
        bps (int): Batches per step
        lhs (str): Tensor name of the left hand side operand
        rhs (str): Tensor name of the right hand side operand
        output (str): Tensor name of the output tensor
        model (Literal['add_model', 'subtract_model'], optional): Model to use.
          Defaults to "add_model".

    Raises:
        ValueError: If an unsupported model is specified
    """
    # Compile graph
    session.prepareDevice()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    # Generate some random input data
    data_shape = [3]
    data_shape.insert(0, bps)
    data_a = np.random.random_sample(data_shape).astype(np.float32)
    data_b = np.random.random_sample(data_shape).astype(np.float32)

    stepio = popart.PyStepIO({lhs: data_a, rhs: data_b}, anchors)
    session.run(stepio)

    if model == "add_model":
        assert np.allclose(anchors[output], data_a + data_b)
    elif model == "subtract_model":
        assert np.allclose(anchors[output], data_a - data_b)
    else:
        raise ValueError(f"Model '{model}' not supported")


def run_model_test(bps: int,
                   opts: Optional[popart.SessionOptions] = None,
                   model: Literal["add_model", "subtract_model"] = "add_model"
                   ) -> popart.InferenceSession:
    """Test that the output is expected for a given model with given options.

    Args:
        bps (int): Batches per step
        opts (Optional[popart.SessionOptions], optional): The options to be used for the session.
          Defaults to None.
        model (Literal['add_model', 'subtract_model'], optional): Model to use.
          Defaults to "add_model".

    Returns:
        popart.InferenceSession: The session used in the test
    """
    with tu.create_test_device() as device:
        session, lhs, rhs, output = create_inference_session(device=device,
                                                             bps=bps,
                                                             opts=opts,
                                                             model=model)
        run_session_and_check_result(session,
                                     bps,
                                     lhs,
                                     rhs,
                                     output,
                                     model=model)
    return session


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
        if 'Starting compilation' in line:
            started_engine_compilation = True
        elif 'Loading serialized PopART executable' in line:
            loaded_poplar_executable = True

    # Assert that we didn't both start a compilation AND load an executable
    assert started_engine_compilation != loaded_poplar_executable
    return not started_engine_compilation


@tu.requires_ipu
def test_manual_save_load(tmp_path: Path,
                          capfd: pytest.CaptureFixture) -> None:
    """
    Perform the tests described below with explicit compilation and exporting of the cache file.

    Test:
    1. That engine caching works for two identical sessions
    2. That the cached engine isn't loaded for a different session

    Args:
        tmp_path (Path): Temporary directory
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
    """
    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel('DEBUG')

    def compile_and_export(bps, filename):
        with tu.create_test_device() as device:
            session, _, _, _ = create_inference_session(device=device, bps=bps)
            assert not os.path.isfile(filename)
            session.compileAndExport(filename)
            assert os.path.isfile(filename)

    def load_and_run(bps, filename):
        with tu.create_test_device() as device:
            session, lhs, rhs, output = create_inference_session(device=device,
                                                                 bps=bps)
            if filename is not None:
                session.loadExecutable(filename)

        run_session_and_check_result(session, bps, lhs, rhs, output)

    executable_path = str(tmp_path / 'model.popart')
    compile_and_export(2, executable_path)
    assert loaded_saved_executable(capfd) is False

    # Check the executable was loaded from the file.
    load_and_run(2, executable_path)
    assert loaded_saved_executable(capfd) is True

    # Check it compiles if we don't load the file.
    load_and_run(2, None)
    assert loaded_saved_executable(capfd) is False


@tu.requires_ipu
def test_simple_cache_hit(tmp_path: Path, capfd: pytest.CaptureFixture):
    """
    Perform the tests described below with the `enableEngineCaching` flag set to True.

    Test:
    1. That engine caching works for two identical sessions
    2. That the cached engine isn't loaded for a different session

    Args:
        tmp_path (Path): Temporary directory
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
    """
    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel('DEBUG')

    opts = popart.SessionOptions()
    opts.enableEngineCaching = True
    opts.cachePath = str(tmp_path / 'saved_graph')

    run_model_test(2, opts)
    assert loaded_saved_executable(capfd) is False

    # Check engine caching works for two identical sessions.
    run_model_test(2, opts)
    assert loaded_saved_executable(capfd) is True

    # Check the cached engine isn't loaded for a different session.
    run_model_test(70, opts)
    assert loaded_saved_executable(capfd) is False


@tu.requires_ipu
def test_cache_miss_on_engine_option_change(
        tmp_path: Path, capfd: pytest.CaptureFixture) -> None:
    """
    Test that no cache is hit if we change engine options affecting the executable between runs.

    Args:
        tmp_path (Path): Temporary directory
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
    """
    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel('DEBUG')

    opts1 = popart.SessionOptions()
    opts1.enableEngineCaching = True
    opts1.cachePath = str(tmp_path / 'saved_graph')
    opts1.engineOptions["opt.enableInlining"] = "false"

    opts2 = popart.SessionOptions()
    opts2.enableEngineCaching = True
    opts2.cachePath = str(tmp_path / 'saved_graph')
    opts2.engineOptions["opt.enableInlining"] = "true"

    run_model_test(2, opts1)
    assert loaded_saved_executable(capfd) is False

    # Check engine caching works for two identical sessions.
    run_model_test(2, opts2)
    assert loaded_saved_executable(capfd) is False


@tu.requires_ipu
@pytest.mark.parametrize("varname", ["POPART_CACHE_DIR", "POPXL_CACHE_DIR"])
def test_cache_environment_variable(tmp_path: Path, monkeypatch: MonkeyPatch,
                                    capfd: pytest.CaptureFixture,
                                    varname: str) -> None:
    """Test caching as enabled via env POPART_CACHE_DIR or POPXL_CACHE_DIR.

    Args:
        tmp_path (Path): Temporary directory
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
        varname (str): Variable name to set as environmental variable
    """
    # Need to activate the logger in order to check whether we are compiling or loading from cache
    popart.getLogger().setLevel('DEBUG')

    monkeypatch.setenv(varname, str(tmp_path / 'saved_graph'))

    opts = popart.SessionOptions()

    run_model_test(2, opts)
    assert loaded_saved_executable(capfd) is False

    # Check engine caching works for two identical sessions.
    run_model_test(2, opts)
    assert loaded_saved_executable(capfd) is True


@tu.requires_ipu
def test_bad_load(tmp_path: Path) -> None:
    """
    Create 2 models with identical stream names, check that the second model doesn't load the first.

    Args:
        tmp_path (Path): Temporary directory
    """
    opts = popart.SessionOptions()
    opts.enableEngineCaching = True
    opts.cachePath = str(tmp_path / 'saved_graph')

    print('Running first model')
    run_model_test(bps=1, opts=opts, model="add_model")
    print()

    print('Running second model')
    run_model_test(bps=1, opts=opts, model="subtract_model")
    print()


@tu.requires_ipu
def test_get_reports(tmp_path: Path) -> None:
    """Test that obtaining a report throws an error when using cached executables.

    Args:
        tmp_path (Path): Temporary directory
    """
    opts = popart.SessionOptions()
    opts.enableEngineCaching = True
    opts.cachePath = str(tmp_path / 'saved_graph')

    run_model_test(bps=1, opts=opts, model="add_model")
    cached_session = run_model_test(bps=1, opts=opts, model="add_model")

    expected_error = 'Unable to get reports when using a cached executable.'

    with pytest.raises(popart.popart_exception) as e_info:
        cached_session.getSummaryReport()
    error = e_info.value.args[0].splitlines()[0]
    assert error == expected_error


def is_stored_restored(capfd: pytest.CaptureFixture) -> Tuple[bool, bool]:
    """Check the log for stores and restores of profile cache

    Args:
        capfd (pytest.CaptureFixture): _description_

    Returns:
        Tuple[bool, bool]:
            1. First bool is True if profiles were stored
            2. Second bool is True if profiles were restored
    """
    _, stderr = capfd.readouterr()
    stored_profile = False
    restored_profile = False
    for line in stderr.splitlines():
        if 'Storing profiles to cache' in line:
            stored_profile = True
        if 'Restoring profiles from cache' in line:
            restored_profile = True
    return stored_profile, restored_profile


@tu.requires_ipu
def test_manual_cached_profiling(tmp_path: Path, monkeypatch: MonkeyPatch,
                                 capfd: pytest.CaptureFixture) -> None:
    """Test that profiling works with cached executables.

    Args:
        tmp_path (Path): Temporary path
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
    """
    # Set the environment variable so that we can capture the output
    monkeypatch.setenv("POPLAR_PROFILER_LOG_LEVEL", "DEBUG")
    popart.getLogger().setLevel('INFO')

    # Set cache and profile directories
    cache_dir = tmp_path / 'saved_graph'
    profile_dir = tmp_path / 'saved_profiles'

    # Enable profiling and caching
    opts = popart.SessionOptions()
    opts.engineOptions["autoReport.directory"] = str(profile_dir)
    opts.engineOptions["autoReport.all"] = "true"
    opts.enableEngineCaching = True
    opts.cachePath = str(cache_dir)
    bps = 1

    executable_path = cache_dir / "model.popef"
    profile_profile_path = profile_dir / "inference" / "profile.pop"
    profile_cache_path = cache_dir / "inference" / "profile.pop"

    with tu.create_test_device() as device:
        session, _, _, _ = create_inference_session(device=device,
                                                    bps=bps,
                                                    opts=opts)
        # The files should not exist by this point
        assert not executable_path.is_file()
        assert not profile_cache_path.is_file()
        assert not profile_profile_path.is_file()

        # After compilation all files should have been created
        session.compileAndExport(str(executable_path))

    # NOTE: The engine object (created by Session) must be destroyed for the
    #       SQLite database to close and for the profile file to be renamed
    #       from profile_<random>.pop to profile.pop
    del session
    assert executable_path.is_file()
    assert profile_cache_path.is_file()
    assert profile_profile_path.is_file()

    # Check that the files have been copied
    stored_profile, restored_profile = is_stored_restored(capfd=capfd)
    assert stored_profile
    assert not restored_profile

    # Remove the original files to ensure that nothing is lingering on from the compilation
    profile_profile_path.unlink()

    with tu.create_test_device() as device:
        session, lhs, rhs, output = create_inference_session(device=device,
                                                             bps=bps,
                                                             opts=opts)
        session.loadExecutable(str(executable_path))
        run_session_and_check_result(session, bps, lhs, rhs, output)

    # NOTE: The engine object (created by Session) must be destroyed for the
    #       SQLite database to close and for the profile file to be renamed
    #       from profile_<random>.pop to profile.pop
    del session
    assert profile_profile_path.is_file()

    # Check that the files have been copied
    stored_profile, restored_profile = is_stored_restored(capfd=capfd)
    assert not stored_profile
    assert restored_profile


@tu.requires_ipu
@pytest.mark.parametrize("cache_env_var",
                         [None, "POPART_CACHE_DIR", "POPXL_CACHE_DIR"])
# TODO: T63870 - enable profiling_env_var True
# @pytest.mark.parametrize("profiling_env_var", [False, True])
@pytest.mark.parametrize("profiling_env_var", [False])
def test_cached_profiling(tmp_path: Path, monkeypatch: MonkeyPatch,
                          cache_env_var: Optional[str],
                          profiling_env_var: bool) -> None:
    """Test profiling of cached executables.

    Test with:
    - Manually setting caching in session options
    - Using environmental variables

    Args:
        tmp_path (Path): Temporary directory
        monkeypatch (MonkeyPatch): MonkeyPatch used for setting the env variables safely
        cache_env_var (Optional[str]): Variable name to set as environmental variable
        profiling_env_var (bool): Whether to specify the profiling as an environmental
            variable or as a session option
    """
    # Set paths
    cache_dir = tmp_path / 'saved_graph'
    profile_dir = tmp_path / 'saved_profiles'
    profile_file = profile_dir.joinpath("inference", "profile.pop")
    cached_profile_file = cache_dir.joinpath("inference", "profile.pop")
    debug_file = cache_dir.joinpath("inference", "debug.cbor")
    cached_debug_file = cache_dir.joinpath("inference", "debug.cbor")

    # NOTE: We must set "POPART_CACHE_DIR" and "POPXL_CACHE_DIR" before
    #       popart.SessionOptions in order for the SessionOption constructor to
    #       set the cache dir correctly
    if cache_env_var is not None:
        monkeypatch.setenv(cache_env_var, str(cache_dir))

    opts = popart.SessionOptions()

    if cache_env_var is None:
        opts.enableEngineCaching = True
        opts.cachePath = str(cache_dir)

    engine_option_dict = {
        "autoReport.directory": str(profile_dir),
        "autoReport.all": "true",
    }

    if profiling_env_var:
        # TODO: T63870 - update environment variable
        #       (will probably not be POPLAR_ENGINE_OPTIONS in the future)
        monkeypatch.setenv("POPLAR_ENGINE_OPTIONS",
                           str(engine_option_dict).replace("'", '"'))
    else:
        for key, val in engine_option_dict.items():
            opts.engineOptions[key] = val

    # Run the model and check that all has gone as expected
    run_model_test(2, opts)
    # Check that the profiling files exist
    assert profile_file.is_file()
    assert cached_profile_file.is_file()
    # Check that only the profile_file has written the execution profile
    report_1_profile = pva.openReport(str(profile_file))
    report_1_cached_profile = pva.openReport(str(cached_profile_file))
    assert report_1_profile.execution.totalCycles.total > 0
    assert report_1_cached_profile.execution.totalCycles.total == 0
    if profiling_env_var:
        assert debug_file.is_file()
        assert cached_debug_file.is_file()
    print("First run was successful")

    # Remove the original files to ensure that nothing is lingering on from the compilation
    profile_file.unlink()
    if profiling_env_var:
        debug_file.unlink()

    # Run the same model again, and check that the profile has been loaded
    run_model_test(2, opts)
    report_2_profile = pva.openReport(str(profile_file))
    report_2_cached_profile = pva.openReport(str(cached_profile_file))
    assert report_2_profile.execution.totalCycles.total > 0
    assert report_2_cached_profile.execution.totalCycles.total == 0
    print("Second run was successful")


def test_implicit_pipelining_custom_fwd_only_cache(tmp_path: Path) -> None:
    """Test if inference within the training session works with caching for explicit pipelining.

    Args:
        tmp_path (Path): Temporary directory
    """
    filename = str(tmp_path / 'model.popart')

    hidden_size = 5
    batches_per_step = 2
    accumulation_factor = 4
    input_shape = [hidden_size, hidden_size]

    data = np.random.normal(0, 0.02,
                            [hidden_size, hidden_size]).astype(np.float32)

    input_data = np.random.normal(
        0, 0.02, [batches_per_step, accumulation_factor] + input_shape).astype(
            np.float32)

    builder = popart.Builder(opsets={
        "ai.onnx": 9,
        "ai.onnx.ml": 1,
        "ai.graphcore": 1
    })

    x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape),
                                  "x_in")

    w0 = builder.addInitializedInputTensor(data, "w0")
    w1 = builder.addInitializedInputTensor(data, "w1")

    with builder.virtualGraph(0), builder.pipelineStage(0):
        o = builder.aiOnnx.mul([x_in, w0])

    with builder.virtualGraph(1), builder.pipelineStage(1):
        o = builder.aiOnnx.mul([o, w1])
        l1 = builder.aiGraphcore.l1loss([o], 0.1)

    proto = builder.getModelProto()

    data_flow = popart.DataFlow(batches_per_step,
                                {l1: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    # Disable outlining to make debugging easier
    opts.enableOutlining = False
    opts.enablePipelining = True
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = accumulation_factor
    opts.autoRecomputation = popart.RecomputationType.Pipeline
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    # Option under test
    opts.createImplicitPipeliningFwdOnlyProgram = True

    pat = popart.Patterns(popart.PatternsLevel.Default)

    with tu.create_test_device(numIpus=4) as device0:
        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=data_flow,
                                         userOptions=opts,
                                         loss=l1,
                                         optimizer=popart.ConstSGD(1),
                                         patterns=pat,
                                         deviceInfo=device0)

        session.prepareDevice()

        # Old session
        session.compileAndExport(filename)

        # New session
        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=data_flow,
                                         userOptions=opts,
                                         loss=l1,
                                         optimizer=popart.ConstSGD(1),
                                         patterns=pat,
                                         deviceInfo=device0)

        session.loadExecutable(filename)

        session.prepareDevice()

        anchors = session.initAnchorArrays()
        inputs = {x_in: input_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.weightsFromHost()

        # Test if both entry points work with a cached executable
        session.run(stepio)
        session.run("implicitPipeliningFwdOnly", stepio)
