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

    dataFlow = poponnx.DataFlow(1, 1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.Session(
        fnModel=proto, dataFeed=dataFlow, outputdir=str(tmpdir))

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

    dataFlow = poponnx.DataFlow(1, 1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.Session(
        fnModel=proto, dataFeed=dataFlow, outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()

    inputs = {i1: np.array([1., 3.], dtype=np.float16)}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.allclose(anchors[o], np.array([3., 7.], dtype=np.float16)))
