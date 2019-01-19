import numpy as np
import pytest
import poponnx
import torch
import test_util as tu


def test_add_fp16(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT16", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.setDevice(tu.get_poplar_cpu_device())
    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()

    inputs = {
        i1: np.array([1., 3.], dtype=np.float16),
        i2: np.array([7., 8.], dtype=np.float16)
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.allclose(anchors[o], np.array([8., 11.], dtype=np.float16)))


def test_add_variable_fp16(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT16", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInitializedInputTensor(
        np.array([2., 4.], dtype=np.float16))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.setDevice(tu.get_poplar_cpu_device())
    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()

    inputs = {i1: np.array([1., 3.], dtype=np.float16)}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.allclose(anchors[o], np.array([3., 7.], dtype=np.float16)))


def test_fp16transpose(tmpdir):
    """
     The compute graph :

      [c0] const f16 [5x6] ---> transpose to [6x5] -|
                                                    |--- add --> 6x5
      [d0] input f16 [1x6] ---> transpose to [6x1] -|

      This test compares poponnx to numpy, inference only

    """

    c0_data = np.random.randn(5, 6).astype(np.float16) * 1
    c0_info = poponnx.TensorInfo("FLOAT16", [5, 6])

    d0_data = np.random.randn(1, 6).astype(np.float16) * 1
    d0_info = poponnx.TensorInfo("FLOAT16", [1, 6])

    builder = poponnx.Builder()
    c0 = builder.constant(c0_data, "c0")
    d0 = builder.addInputTensor(d0_info)

    c0_t = builder.transpose([c0], [], "c0_transpose")
    d0_t = builder.transpose([d0], [], "d0_transpose")

    o = builder.add([d0_t, c0_t], "add")
    builder.addOutputTensor(o)

    proto = builder.getModelProto()
    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})
    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)
    session.setDevice(tu.get_poplar_cpu_device())
    session.prepareDevice()
    anchors = session.initAnchorArrays()
    inputs = {d0: d0_data}
    stepio = poponnx.PyStepIO(inputs, anchors)
    session.infer(stepio)

    # numpy reference
    npout = np.transpose(c0_data) + np.transpose(d0_data)

    # lowering default tolerance for half-precision
    assert (np.allclose(anchors[o], npout, atol=1e-4, rtol=1e-2))
