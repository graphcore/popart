# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import test_util as tu


def identity_inference_session(tmpdir, inputShape, inputArray, BPS, art, R=1):

    builder = popart.Builder()

    inInfo = popart.TensorInfo("FLOAT", inputShape)

    i1 = builder.addInputTensor(inInfo)
    o = builder.aiOnnx.identity([i1])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    batchesPerStep = BPS
    dataFlow = popart.DataFlow(batchesPerStep, {o: art})

    opts = popart.SessionOptions()
    opts.replicatedGraphCount = R
    opts.enableReplicatedGraphs = R > 1
    # TODO: Remove when T14730 is resolved.
    opts.enablePrefetchDatastreams = R < 1

    device = tu.create_test_device(numIpus=R)

    session = popart.InferenceSession(fnModel=proto,
                                      dataFeed=dataFlow,
                                      deviceInfo=device,
                                      userOptions=opts)

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {
        i1: np.array(inputArray, dtype=np.float32),
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    return anchors[o]


def simple_training_session(tmpdir, inputShape, inputArray, BPS, art, GA=1):

    builder = popart.Builder()

    inInfo = popart.TensorInfo("FLOAT", inputShape)

    i1 = builder.addInputTensor(inInfo)
    w1 = builder.addInitializedInputTensor(
        np.zeros(inputShape, dtype=np.float32), "w1")
    o = builder.aiOnnx.add([i1, w1])
    l1 = builder.aiGraphcore.l1loss([o], 0.0)

    loss = popart.IdentityLoss(l1, "idLossVal")

    proto = builder.getModelProto()

    batchesPerStep = BPS
    dataFlow = popart.DataFlow(batchesPerStep, {o: art})

    opts = popart.SessionOptions()
    opts.accumulationFactor = GA
    opts.enableGradientAccumulation = GA > 1

    session = popart.TrainingSession(fnModel=proto,
                                     dataFeed=dataFlow,
                                     deviceInfo=tu.create_test_device(),
                                     userOptions=opts,
                                     losses=[loss],
                                     optimizer=popart.ConstSGD(0.01))

    session.prepareDevice()
    session.weightsFromHost()

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
    art = popart.AnchorReturnType("All")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_evern1(tmpdir):
    inputArray = 1
    art = popart.AnchorReturnType("EveryN", 1)
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_final1(tmpdir):
    inputArray = 1
    art = popart.AnchorReturnType("Final")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_sum1(tmpdir):
    inputArray = 1
    art = popart.AnchorReturnType("Sum")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


# 0-d input tensors, batchesPerStep > 1
def test_returntype_all2(tmpdir):
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    art = popart.AnchorReturnType("All")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 8, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_everyn2(tmpdir):
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    art = popart.AnchorReturnType("EveryN", 4)
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 8, art)
    assert (np.array_equal(anchors_o, [4, 8]))


def test_returntype_final2(tmpdir):
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    art = popart.AnchorReturnType("Final")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 8, art)
    assert (np.array_equal(anchors_o, 8))


def test_returntype_sum2(tmpdir):
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    art = popart.AnchorReturnType("Sum")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 8, art)
    assert (np.array_equal(anchors_o, 36))


# 1-d input tensors, batchesPerStep = 1
def test_returntype_all3(tmpdir):
    inputArray = [1, 2]
    art = popart.AnchorReturnType("All")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_everyn3(tmpdir):
    inputArray = [1, 2]
    art = popart.AnchorReturnType("EveryN", 1)
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_final3(tmpdir):
    inputArray = [1, 2]
    art = popart.AnchorReturnType("Final")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_sum3(tmpdir):
    inputArray = [1, 2]
    art = popart.AnchorReturnType("Sum")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 1, art)
    assert (np.array_equal(anchors_o, inputArray))


# 1-d input tensors, batchesPerStep > 1
def test_returntype_all4(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("All")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 4, art)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_everyn4(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("EveryN", 2)
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 4, art)
    assert (np.array_equal(anchors_o, [[3, 4], [7, 8]]))


def test_returntype_final4(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("Final")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 4, art)
    assert (np.array_equal(anchors_o, [7, 8]))


def test_returntype_sum4(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("Sum")
    anchors_o = identity_inference_session(tmpdir, [2], inputArray, 4, art)
    assert (np.array_equal(anchors_o, [16, 20]))


# 0-d input tensors, batchesPerStep > 1, replication = 2
@tu.requires_ipu
def test_returntype_all5(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("All")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 4, art, R=2)
    assert (np.array_equal(anchors_o, inputArray))


@tu.requires_ipu
def test_returntype_everyn5(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("EveryN", 2)
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 4, art, R=2)
    assert (np.array_equal(anchors_o, [[3, 4], [7, 8]]))


@tu.requires_ipu
def test_returntype_final5(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("Final")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 4, art, R=2)
    assert (np.array_equal(anchors_o, [7, 8]))


@tu.requires_ipu
def test_returntype_sum5(tmpdir):
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    art = popart.AnchorReturnType("Sum")
    anchors_o = identity_inference_session(tmpdir, [], inputArray, 4, art, R=2)
    assert (np.array_equal(anchors_o, [16, 20]))


# 1-d input tensors, batchesPerStep > 1, gradient accumulation = 2
def test_returntype_all6(tmpdir):
    inputArray = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    art = popart.AnchorReturnType("All")
    anchors_o = simple_training_session(tmpdir, [2], inputArray, 2, art, GA=2)
    assert (np.array_equal(anchors_o, inputArray))


def test_returntype_everyn6(tmpdir):
    inputArray = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    art = popart.AnchorReturnType("EveryN", 2)
    anchors_o = simple_training_session(tmpdir, [2], inputArray, 2, art, GA=2)
    assert (np.array_equal(anchors_o, [[3, 4], [7, 8]]))


def test_returntype_final6(tmpdir):
    inputArray = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    art = popart.AnchorReturnType("Final")
    anchors_o = simple_training_session(tmpdir, [2], inputArray, 2, art, GA=2)
    assert (np.array_equal(anchors_o, [7, 8]))


def test_returntype_sum6(tmpdir):
    inputArray = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    art = popart.AnchorReturnType("Sum")
    anchors_o = simple_training_session(tmpdir, [2], inputArray, 2, art, GA=2)
    assert (np.array_equal(anchors_o, [16, 20]))


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
        popart.AnchorReturnType("EveryN", -1)

    assert (e_info.value.args[0] ==
            "Anchor return period must be greater than zero")


def test_invalid_period3():
    with pytest.raises(popart.popart_exception) as e_info:
        popart.AnchorReturnType("EveryN")

    assert (e_info.value.args[0] ==
            "Must specify return period with option 'EVERYN'")
