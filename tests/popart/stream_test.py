import numpy as np
import pytest
import popart
import torch
import test_util as tu


def test_stream_on_off(tmpdir):

    builder = popart.Builder()
    shape = popart.TensorInfo("FLOAT16", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(
        1, {
            i1: popart.AnchorReturnType("ALL"),
            i2: popart.AnchorReturnType("ALL"),
            o: popart.AnchorReturnType("ALL")
        })

    session = popart.InferenceSession(fnModel=proto,
                                       dataFeed=dataFlow,
                                       deviceInfo=tu.get_poplar_cpu_device())

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {
        i1: np.array([1., 3.], dtype=np.float16),
        i2: np.array([7., 8.], dtype=np.float16)
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    # confirm that writing device-to-host of a Stream Tensor returns correctly (unchanged)
    assert (np.allclose(anchors[i1], np.array([1., 3.], dtype=np.float16)))
    assert (np.allclose(anchors[i2], np.array([7., 8.], dtype=np.float16)))
