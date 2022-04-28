# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import os
import numpy as np
import popart
import pytest
import onnx

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu
def test_simple_save_load(tmp_path, capfd):
    """
    Test:
    1. That engine caching works for two identical sessions
    2. That the cached engine isn't loaded for a different session
    """

    def _init_session(bps, device):
        # Create a builder and construct a graph
        builder = popart.Builder()

        data_shape = [3]
        data_info = popart.TensorInfo("FLOAT", data_shape)

        a = builder.addInputTensor(data_info)
        b = builder.addInputTensor(data_info)

        o = builder.aiOnnx.add([a, b])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        # Describe how to run the model
        dataFlow = popart.DataFlow(bps, {o: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()

        # Create a session to compile and execute the graph
        return popart.InferenceSession(fnModel=proto,
                                       dataFlow=dataFlow,
                                       userOptions=opts,
                                       deviceInfo=device), a, b, o

    def compile_and_export(bps, filename):
        with tu.create_test_device() as device:
            session, _, _, _ = _init_session(bps, device)
            assert not os.path.isfile(filename)
            session.compileAndExport(filename)
            assert os.path.isfile(filename)

    def load_and_run(bps, filename):
        with tu.create_test_device() as device:
            session, a, b, o = _init_session(bps, device)
            if filename is not None:
                session.loadExecutable(filename)

            # Compile graph
            session.prepareDevice()

            # Create buffers to receive results from the execution
            anchors = session.initAnchorArrays()

            # Generate some random input data
            data_shape = [3]
            data_shape.insert(0, bps)
            data_a = np.random.random_sample(data_shape).astype(np.float32)
            data_b = np.random.random_sample(data_shape).astype(np.float32)

            stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
            session.run(stepio)

        assert np.allclose(anchors[o], data_a + data_b)

    # Check the log output to see if an engine was compiled,
    # or if a cached engine was used.
    def loaded_saved_executable():
        _, stderr = capfd.readouterr()
        startedEngineCompilation = False
        loadedPoplarExecutable = False
        for line in stderr.splitlines():
            if 'Starting compilation' in line:
                startedEngineCompilation = True
            elif 'Loading serialized PopART executable' in line:
                loadedPoplarExecutable = True

        assert startedEngineCompilation != loadedPoplarExecutable
        return not startedEngineCompilation

    popart.getLogger().setLevel('DEBUG')
    exeFile = str(tmp_path / 'model.popart')
    compile_and_export(2, exeFile)
    assert loaded_saved_executable() is False

    # Check the executable was loaded from the file.
    load_and_run(2, exeFile)
    assert loaded_saved_executable() is True

    # Check it compiles if we don't load the file.
    load_and_run(2, None)
    assert loaded_saved_executable() is False


def run_session(bps, opts):
    with tu.create_test_device() as device:

        # Create a builder and construct a graph
        builder = popart.Builder()

        data_shape = [3]
        data_info = popart.TensorInfo("FLOAT", data_shape)

        a = builder.addInputTensor(data_info)
        b = builder.addInputTensor(data_info)

        o = builder.aiOnnx.add([a, b])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        # Describe how to run the model
        dataFlow = popart.DataFlow(bps, {o: popart.AnchorReturnType("All")})

        # Create a session to compile and execute the graph
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        # Compile graph
        session.prepareDevice()

        # Create buffers to receive results from the execution
        anchors = session.initAnchorArrays()

        # Generate some random input data
        data_shape.insert(0, bps)
        data_a = np.random.random_sample(data_shape).astype(np.float32)
        data_b = np.random.random_sample(data_shape).astype(np.float32)

        stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
        session.run(stepio)

        assert np.allclose(anchors[o], data_a + data_b)


# Check the log output to see if an engine was compiled,
# or if a cached engine was used.
def loaded_saved_executable(capfd):
    _, stderr = capfd.readouterr()
    startedEngineCompilation = False
    loadedPoplarExecutable = False
    for line in stderr.splitlines():
        if 'Starting compilation' in line:
            startedEngineCompilation = True
        elif 'Loading serialized PopART executable' in line:
            loadedPoplarExecutable = True

    assert startedEngineCompilation != loadedPoplarExecutable
    return not startedEngineCompilation


@tu.requires_ipu
def test_simple_cache_hit(tmp_path, capfd):
    """
    Test:
    1. That engine caching works for two identical sessions
    2. That the cached engine isn't loaded for a different session
    """
    popart.getLogger().setLevel('DEBUG')

    opts = popart.SessionOptions()
    opts.enableEngineCaching = True
    opts.cachePath = str(tmp_path / 'saved_graph')

    run_session(2, opts)
    assert loaded_saved_executable(capfd) is False

    # Check engine caching works for two identical sessions.
    run_session(2, opts)
    assert loaded_saved_executable(capfd) is True

    # Check the cached engine isn't loaded for a different session.
    run_session(70, opts)
    assert loaded_saved_executable(capfd) is False


@tu.requires_ipu
def test_cache_miss_on_engine_option_change(tmp_path, capfd):
    """ Test that if we change engine options that affect the executable between
    runs then we don't get a cache hit. """
    popart.getLogger().setLevel('DEBUG')

    opts1 = popart.SessionOptions()
    opts1.enableEngineCaching = True
    opts1.cachePath = str(tmp_path / 'saved_graph')
    opts1.engineOptions["opt.enableInlining"] = "false"

    opts2 = popart.SessionOptions()
    opts2.enableEngineCaching = True
    opts2.cachePath = str(tmp_path / 'saved_graph')
    opts2.engineOptions["opt.enableInlining"] = "true"

    run_session(2, opts1)
    assert loaded_saved_executable(capfd) is False

    # Check engine caching works for two identical sessions.
    run_session(2, opts2)
    assert loaded_saved_executable(capfd) is False


@tu.requires_ipu
@pytest.mark.parametrize("varname", ["POPART_CACHE_DIR", "POPXL_CACHE_DIR"])
def test_cache_environment_variable(tmp_path, capfd, varname):
    """
    Test caching as enabled via env POPART_CACHE_DIR or POPXL_CACHE_DIR
    """
    popart.getLogger().setLevel('DEBUG')
    os.environ[varname] = str(tmp_path / 'saved_graph')

    opts = popart.SessionOptions()

    run_session(2, opts)
    assert loaded_saved_executable(capfd) is False

    # Check engine caching works for two identical sessions.
    run_session(2, opts)
    assert loaded_saved_executable(capfd) is True

    del os.environ[varname]


@tu.requires_ipu
def test_bad_load(tmp_path):
    """
    Create 2 models with identical stream names
    """

    def get_add_model():
        # Create a builder and construct a graph
        builder = popart.Builder()

        data_shape = popart.TensorInfo("FLOAT", [1])

        a = builder.addInputTensor(data_shape)
        b = builder.addInputTensor(data_shape)

        o = builder.aiOnnx.add([a, b])
        o = builder.aiOnnx.identity([o])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        return proto, a, b, o

    def get_sub_model():
        # Create a builder and construct a graph
        builder = popart.Builder()

        data_shape = popart.TensorInfo("FLOAT", [1])

        a = builder.addInputTensor(data_shape)
        b = builder.addInputTensor(data_shape)

        o = builder.aiOnnx.sub([a, b])
        o = builder.aiOnnx.identity([o])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        return proto, a, b, o

    def run_test(proto, a, b, o, test):
        with tu.create_test_device() as device:

            # Describe how to run the model
            dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

            opts = popart.SessionOptions()
            opts.enableEngineCaching = True
            opts.cachePath = str(tmp_path / 'saved_graph')

            # Create a session to compile and execute the graph
            session = popart.InferenceSession(fnModel=proto,
                                              dataFlow=dataFlow,
                                              userOptions=opts,
                                              deviceInfo=device)

            # Compile graph
            session.prepareDevice()

            # Create buffers to receive results from the execution
            anchors = session.initAnchorArrays()

            # Generate some random input data
            data_a = np.random.rand(1).astype(np.float32)
            data_b = np.random.rand(1).astype(np.float32)

            stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
            session.run(stepio)

            assert test(data_a, data_b, anchors[o])

    print('Running first model')
    run_test(*get_add_model(), lambda a, b, c: c == a + b)
    print()
    print('Running second model')
    run_test(*get_sub_model(), lambda a, b, c: c == a - b)
    print()


@tu.requires_ipu
def test_get_reports(tmp_path):
    def run_session(device):
        # Create a builder and construct a graph
        builder = popart.Builder()

        data_shape = popart.TensorInfo("FLOAT", [1])

        a = builder.addInputTensor(data_shape)
        b = builder.addInputTensor(data_shape)

        o = builder.aiOnnx.add([a, b])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        # Describe how to run the model
        dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        opts.enableEngineCaching = True
        opts.cachePath = str(tmp_path / 'saved_graph')

        # Create a session to compile and execute the graph
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        # Compile graph
        session.prepareDevice()

        # Create buffers to receive results from the execution
        anchors = session.initAnchorArrays()

        # Generate some random input data
        data_a = np.random.rand(1).astype(np.float32)
        data_b = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
        session.run(stepio)

        return session

    with tu.create_test_device() as device:
        run_session(device)
    with tu.create_test_device() as device:
        cached_session = run_session(device)

    expected_error = 'Unable to get reports when using a cached executable.'

    with pytest.raises(popart.popart_exception) as e_info:
        cached_session.getSummaryReport()
    error = e_info.value.args[0].splitlines()[0]
    assert error == expected_error


def test_implicit_pipelining_custom_fwd_only_cache(tmpdir):
    """Test if running inference within the training session works
    with caching
    """
    filename = str(tmpdir / 'model.popart')

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

    dataFlow = popart.DataFlow(batches_per_step,
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
                                         dataFlow=dataFlow,
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
                                         dataFlow=dataFlow,
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
