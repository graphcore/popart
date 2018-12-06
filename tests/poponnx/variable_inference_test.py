import numpy as np
import pytest
import poponnx
import torch
import test_util as tu


def inference_add_to_variable(tmpdir, type, np_type):
    builder = poponnx.Builder()

    shape = poponnx.TensorInfo(type, [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInitializedInputTensor(np.array([2., 4.], dtype=np_type))
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

    inputs = {i1: np.array([1., 3.], dtype=np_type)}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.allclose(anchors[o], np.array([3., 7.], dtype=np_type)))


def test_add_variable_fp32(tmpdir):
    inference_add_to_variable(tmpdir, "FLOAT", np.float32)


def test_add_variable_fp16(tmpdir):
    inference_add_to_variable(tmpdir, "FLOAT16", np.float16)
