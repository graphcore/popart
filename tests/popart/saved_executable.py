# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import time
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# Test:
# 1. That engine caching works for two identical sessions
# 2. That the cached engine isn't loaded for a different session
@tu.requires_ipu
def test_simple_load(tmp_path):
    def run_session(bps):
        device = tu.create_test_device()

        start = time.clock()

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
        dataFlow = popart.DataFlow(bps, {o: popart.AnchorReturnType("ALL")})

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
        data_shape.insert(0, bps)
        data_a = np.random.random_sample(data_shape).astype(np.float32)
        data_b = np.random.random_sample(data_shape).astype(np.float32)

        stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
        session.run(stepio)

        assert np.allclose(anchors[o], data_a + data_b)
        return time.clock() - start

    first_duration = run_session(2)
    second_duration = run_session(2)
    third_duration = run_session(70)

    # There is no direct way to test whether the cached executable was used,
    # but using the cached graph should be at least twice as fast as not.
    assert (first_duration / 2) > second_duration  # 1.
    assert (first_duration / 2) < third_duration  # 2.


# create 2 models with identical stream names
@tu.requires_ipu
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
        device = tu.create_test_device()

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


@tu.requires_ipu
def test_get_reports(tmp_path):
    def run_session():
        device = tu.create_test_device()

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
