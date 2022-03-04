# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import os
import numpy as np
import popart
import pytest

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
def test_cache_environment_variable(tmp_path, capfd):
    """
    Test chaching enabled via env POPART_CACHE_DIR
    """
    popart.getLogger().setLevel('DEBUG')
    os.environ['POPART_CACHE_DIR'] = str(tmp_path / 'saved_graph')

    opts = popart.SessionOptions()

    run_session(2, opts)
    assert loaded_saved_executable(capfd) is False

    # Check engine caching works for two identical sessions.
    run_session(2, opts)
    assert loaded_saved_executable(capfd) is True

    del os.environ['POPART_CACHE_DIR']


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
