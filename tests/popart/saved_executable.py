import numpy as np
import popart
import time
import pytest


def test_simple_load(tmp_path):
    def run_session():
        device = popart.DeviceManager().acquireAvailableDevice()
        if not device:
            pytest.skip("Unable to acquire device")

        start = time.clock()

        # Create a builder and construct a graph
        builder = popart.Builder()

        data_shape = popart.TensorInfo("FLOAT", [1])

        a = builder.addInputTensor(data_shape)
        b = builder.addInputTensor(data_shape)

        o = builder.aiOnnx.add([a, b])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        # Describe how to run the model
        dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

        opts = popart.SessionOptions()
        opts.enableEngineCaching = True
        opts.cachePath = str(tmp_path / 'saved_graph')

        # Create a session to compile and execute the graph
        session = popart.InferenceSession(fnModel=proto,
                                          dataFeed=dataFlow,
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

        assert anchors[o] == data_a + data_b

        return time.clock() - start

    first_duration = run_session()
    second_duration = run_session()
    # There is no direct way to test whether the cached executable was used,
    # but using the cached graph should be at least twice as fast as not.
    assert (first_duration / 2) > second_duration


# Check that no error is thrown is opts.cachePath is set and the device is a cpu device.
def test_cpu_device(tmp_path):
    # Create a builder and construct a graph
    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT", [1])

    a = builder.addInputTensor(data_shape)
    b = builder.addInputTensor(data_shape)

    o = builder.aiOnnx.add([a, b])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    # Describe how to run the model
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    opts = popart.SessionOptions()
    opts.enableEngineCaching = True
    opts.cachePath = str(tmp_path / 'saved_graph')

    # Create a session to compile and execute the graph
    session = popart.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        userOptions=opts,
        deviceInfo=popart.DeviceManager().createCpuDevice())

    # Compile graph
    session.prepareDevice()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    # Generate some random input data
    data_a = np.random.rand(1).astype(np.float32)
    data_b = np.random.rand(1).astype(np.float32)

    stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
    session.run(stepio)

    assert anchors[o] == data_a + data_b


# create 2 models with identical stream names
def test_bad_load(tmp_path):
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
        device = popart.DeviceManager().acquireAvailableDevice()
        if not device:
            pytest.skip("Unable to acquire device")

        # Describe how to run the model
        dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

        opts = popart.SessionOptions()
        opts.enableEngineCaching = True
        opts.cachePath = str(tmp_path / 'saved_graph')

        # Create a session to compile and execute the graph
        session = popart.InferenceSession(fnModel=proto,
                                          dataFeed=dataFlow,
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


def test_get_reports(tmp_path):
    def run_session():
        device = popart.DeviceManager().acquireAvailableDevice()
        if not device:
            pytest.skip("Unable to acquire device")

        # Create a builder and construct a graph
        builder = popart.Builder()

        data_shape = popart.TensorInfo("FLOAT", [1])

        a = builder.addInputTensor(data_shape)
        b = builder.addInputTensor(data_shape)

        o = builder.aiOnnx.add([a, b])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        # Describe how to run the model
        dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

        opts = popart.SessionOptions()
        opts.enableEngineCaching = True
        opts.cachePath = str(tmp_path / 'saved_graph')

        # Create a session to compile and execute the graph
        session = popart.InferenceSession(fnModel=proto,
                                          dataFeed=dataFlow,
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

    run_session()
    cached_session = run_session()

    expected_error = 'Unable to get reports when using a cached executable.'

    with pytest.raises(popart.popart_exception) as e_info:
        cached_session.getGraphReport()
    error = e_info.value.args[0].splitlines()[0]
    assert error == expected_error

    with pytest.raises(popart.popart_exception) as e_info:
        cached_session.getSummaryReport()
    error = e_info.value.args[0].splitlines()[0]
    assert error == expected_error

    with pytest.raises(popart.popart_exception) as e_info:
        cached_session.getExecutionReport()
    error = e_info.value.args[0].splitlines()[0]
    assert error == expected_error
