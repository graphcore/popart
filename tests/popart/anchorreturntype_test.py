import numpy as np
import pytest
import popart
import torch
import test_util as tu


def identity_inference_session(tmpdir, inputShape, inputArray, BPS, art):

    builder = popart.Builder()

    inInfo = popart.TensorInfo("FLOAT", inputShape)

    i1 = builder.addInputTensor(inInfo)
    o = builder.aiOnnx.identity([i1])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    batchesPerStep = BPS
    dataFlow = popart.DataFlow(batchesPerStep, {o: art})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFeed=dataFlow,
                                      deviceInfo=tu.get_poplar_cpu_device())

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {
        i1: np.array(inputArray, dtype=np.float32),
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    return anchors[o]


# 0-d input tensors, batchesPerStep = 1
def test_returntype_all1(tmpdir):
    inputArray = 1
    art = popart.AnchorReturnType("ALL")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_evern1(tmpdir):
    inputArray = 1
    art = popart.AnchorReturnType("EVERYN", 1)
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_final1(tmpdir):
    inputArray = 1
    art = popart.AnchorReturnType("FINAL")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


# 0-d input tensors, batchesPerStep > 1
def test_returntype_all2(tmpdir):
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    art = popart.AnchorReturnType("ALL")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 8, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_everyn2(tmpdir):
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    art = popart.AnchorReturnType("EVERYN", 4)
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 8, art)
    assert (np.array_equal(anchors_o, [4, 8]))


def test_returntype_final2(tmpdir):
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    art = popart.AnchorReturnType("FINAL")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 8, art)
    assert (np.array_equal(anchors_o, 8))


# 1-d input tensors, batchesPerStep = 1
def test_returntype_all3(tmpdir):
    inputArray = [1, 2]
    art = popart.AnchorReturnType("ALL")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_everyn3(tmpdir):
    inputArray = [1, 2]
    art = popart.AnchorReturnType("EVERYN", 1)
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_final3(tmpdir):
    inputArray = [1, 2]
    art = popart.AnchorReturnType("FINAL")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


# 1-d input tensors, batchesPerStep > 1
def test_returntype_all4(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("ALL")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 4, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_everyn4(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("EVERYN", 2)
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 4, art)
    assert (np.array_equal(anchors_o, [[3, 4], [7, 8]]))


def test_returntype_final4(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("FINAL")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 4, art)
    assert (np.array_equal(anchors_o, [7, 8]))


# Error cases
def test_invalid_art_id():
    with pytest.raises(popart.popart_exception) as e_info:
        popart.AnchorReturnType("INVALID")

    assert (e_info.value.args[0] ==
            "Invalid anchor return type ID supplied: INVALID")


def test_invalid_period1():
    with pytest.raises(popart.popart_exception) as e_info:
        popart.AnchorReturnType("ALL", 2)

    assert (
        e_info.value.args[0] ==
        "A return period should not be supplied for this anchor return type")


def test_invalid_period2():
    with pytest.raises(popart.popart_exception) as e_info:
        popart.AnchorReturnType("EVERYN", -1)

    assert (e_info.value.args[0] ==
            "Anchor return period must be greater than zero")


def test_invalid_period3():
    with pytest.raises(popart.popart_exception) as e_info:
        popart.AnchorReturnType("EVERYN")

    assert (e_info.value.args[0] ==
            "Must specify return period with option 'EVERYN'")
