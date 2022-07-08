# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import time

# importing test_session and test_util requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu
def test_stepio_bufferinput_ipu():

    builder = popart.Builder()
    shape = popart.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    batches_per_step = 2

    dataFlow = popart.DataFlow(
        batches_per_step,
        {
            i1: popart.AnchorReturnType("All"),
            i2: popart.AnchorReturnType("All"),
            o: popart.AnchorReturnType("All"),
        },
    )

    with tu.create_test_device(1) as device:
        session = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, deviceInfo=device
        )

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        i1_data = np.random.rand(batches_per_step, 2).astype(np.float32)
        i2_data = np.random.rand(batches_per_step, 2).astype(np.float32)

        inputs = {i1: i1_data, i2: i2_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)


@tu.requires_ipu
def test_stepio_callbackinput_ipu():

    builder = popart.Builder()
    shape = popart.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    batches_per_step = 2

    dataFlow = popart.DataFlow(
        batches_per_step,
        {
            i1: popart.AnchorReturnType("All"),
            i2: popart.AnchorReturnType("All"),
            o: popart.AnchorReturnType("All"),
        },
    )

    with tu.create_test_device(1) as device:
        session = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, deviceInfo=device
        )

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        i1_data = np.random.rand(batches_per_step, 2).astype(np.float32)
        i2_data = np.random.rand(batches_per_step, 2).astype(np.float32)

        inputs = {i1: i1_data, i2: i2_data}

        i1_c = 0
        i2_c = 0

        def input_callback(id, prefetch):
            nonlocal i1_c, i2_c

            if prefetch:
                return None

            time.sleep(1)
            print("input_callback ", id)

            t = inputs[id]

            result = None

            print(t)

            if id == i1:
                print("input_callback ", id, len(t), i1_c)
                if i1_c < len(t):
                    result = t[i1_c]
                    i1_c = i1_c + 1

            if id == i2:
                print("input_callback ", id, len(t), i2_c)
                if i2_c < len(t):
                    result = t[i2_c]
                    i2_c = i2_c + 1

            print(result)

            return result

        def input_complete_callback(id):
            print("output_complete_callback ", id)

        i1_d = 0
        i2_d = 0
        o_d = 0

        def output_callback(id):
            nonlocal i1_d, i2_d, o_d

            time.sleep(1)
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

        stepio = popart.PyStepIOCallback(
            input_callback,
            input_complete_callback,
            output_callback,
            output_complete_callback,
        )

        session.run(stepio)

        # confirm that writing device-to-host of a Stream Tensor returns correctly (unchanged)
        assert np.allclose(anchors[i1], i1_data)
        assert np.allclose(anchors[i2], i2_data)

        expected_result = i1_data + i2_data
        assert np.allclose(anchors[o], expected_result)
