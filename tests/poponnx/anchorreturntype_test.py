import numpy as np
import pytest
import poponnx
import torch
import test_util as tu


def identity_inference_session(tmpdir, inputArray, art):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1, 1])

    i1 = builder.addInputTensor(shape)
    o = builder.identity([i1])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(8, 1, {o: art})

    session = poponnx.Session(
        fnModel=proto, dataFeed=dataFlow, outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {
        i1: np.array(inputArray, dtype=np.float32),
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    return anchors[o]


def test_returntype_all(tmpdir):
    inputArray = [[1], [2], [3], [4], [5], [6], [7], [8]]
    art = poponnx.AnchorReturnType("ALL")
    anchors_o = identity_inference_session(tmpdir, inputArray, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_final(tmpdir):
    inputArray = [[1], [2], [3], [4], [5], [6], [7], [8]]
    art = poponnx.AnchorReturnType("FINAL")
    anchors_o = identity_inference_session(tmpdir, inputArray, art)
    assert (np.array_equal(anchors_o, [[8]]))


def test_returntype_everyn(tmpdir):
    inputArray = [[1], [2], [3], [4], [5], [6], [7], [8]]
    art = poponnx.AnchorReturnType("EVERYN", 4)
    anchors_o = identity_inference_session(tmpdir, inputArray, art)
    assert (np.array_equal(anchors_o, [[4], [8]]))


def test_invalid_art_id():
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.AnchorReturnType("INVALID")

    assert (e_info.value.args[0] ==
            "Invalid anchor return type ID supplied: INVALID")


def test_invalid_freq1():
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.AnchorReturnType("ALL", 2)

    assert (
        e_info.value.args[0] ==
        "A return frequency should not be supplied for this anchor return type"
    )


def test_invalid_freq2():
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.AnchorReturnType("EVERYN", -1)

    assert (e_info.value.args[0] ==
            "Anchor return frequency must be greater than zero")


def test_invalid_freq3():
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.AnchorReturnType("EVERYN")

    assert (e_info.value.args[0] ==
            "Must specify return frequency with option 'EVERYN'")
