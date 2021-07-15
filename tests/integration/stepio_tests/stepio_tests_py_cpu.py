# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import time

# importing test_session and test_util requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession
import test_util as tu


def test_stepio_bufferinput():

    builder = popart.Builder()
    shape = popart.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    batches_per_step = 2

    dataFlow = popart.DataFlow(
        batches_per_step, {
            i1: popart.AnchorReturnType("All"),
            i2: popart.AnchorReturnType("All"),
            o: popart.AnchorReturnType("All")
        })

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    i1_data = np.random.rand(batches_per_step, 2).astype(np.float32)
    i2_data = np.random.rand(batches_per_step, 2).astype(np.float32)

    inputs = {i1: i1_data, i2: i2_data}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    return

    # confirm that writing device-to-host of a Stream Tensor returns correctly (unchanged)
    print(i1_data)
    print(anchors[i1])
    assert (np.allclose(anchors[i1], i1_data))

    print(i2_data)
    print(anchors[i2])
    assert (np.allclose(anchors[i2], i2_data))

    expected_result = i1_data + i2_data

    print(expected_result)
    print(anchors[o])

    assert (np.allclose(anchors[o], expected_result))


def test_stepio_callbackinput():

    builder = popart.Builder()
    shape = popart.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    batches_per_step = 2

    dataFlow = popart.DataFlow(
        batches_per_step, {
            i1: popart.AnchorReturnType("All"),
            i2: popart.AnchorReturnType("All"),
            o: popart.AnchorReturnType("All")
        })

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    i1_data = np.random.rand(batches_per_step, 2).astype(np.float32)
    i2_data = np.random.rand(batches_per_step, 2).astype(np.float32)

    inputs = {i1: i1_data, i2: i2_data}

    i1_c = 0
    i2_c = 0

    def input_callback(id, prefetch):
        nonlocal i1_c, i2_c

        time.sleep(2)
        print("input_callback ", id)

        t = inputs[id]

        print(t)

        if id == i1:
            print("input_callback ", id, len(t))
            if (i1_c < len(t)):
                result = t[i1_c]
                i1_c = i1_c + 1

        if id == i2:
            print("input_callback ", id, len(t))
            if (i2_c < len(t)):
                result = t[i2_c]
                i2_c = i2_c + 1

        print(result)

        return result

    def input_complete_callback(id):
        print("input_complete_callback ", id)

    i1_d = 0
    i2_d = 0
    o_d = 0

    def output_callback(id):
        nonlocal i1_d, i2_d, o_d

        time.sleep(2)
        print("output_callback ", id)

        t = anchors[id]

        if id == i1:
            result = t[i1_d]
            i1_d = i1_d + 1

        if id == i2:
            result = t[i2_d]
            i2_d = i2_d + 1

        if id == o:
            result = t[o_d]
            o_d = o_d + 1

        return result

    def output_complete_callback(id):
        print("output_complete_callback ", id)

    stepio = popart.PyStepIOCallback(input_callback, input_complete_callback,
                                     output_callback, output_complete_callback)

    session.run(stepio)

    # confirm that writing device-to-host of a Stream Tensor returns correctly (unchanged)
    assert (np.allclose(anchors[i1], i1_data))
    assert (np.allclose(anchors[i2], i2_data))

    expected_result = i1_data + i2_data
    assert (np.allclose(anchors[o], expected_result))


def test_steio_correct_inputs():
    builder = popart.Builder()
    in0 = builder.addInputTensor("FLOAT", [4])
    sqrt = builder.aiOnnx.sqrt([in0])

    data0 = np.array([1, 4, 9, 16]).astype(np.float32)

    s = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        deviceInfo=popart.DeviceManager().createCpuDevice(),
        dataFlow=popart.DataFlow(1, [sqrt]))
    s.prepareDevice()
    anchors = s.initAnchorArrays()

    stepio0 = popart.PyStepIO({in0: data0, "foo": np.zeros(4)}, anchors)
    stepio1 = popart.PyStepIO({in0: data0, sqrt: np.zeros(4)}, anchors)
    # s.run(stepio0)  # fails, as expected
    with pytest.raises(popart.popart_exception) as e_info:
        s.run(stepio1)
        assert e_info.value.args[0].startswith(
            f"The tensor '{sqrt}'  has been provided as one of the inputs to the stepIO"
        )
