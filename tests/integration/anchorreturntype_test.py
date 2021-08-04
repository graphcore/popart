# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import test_util as tu

import popart

ALL = popart.AnchorReturnType("all")
FINAL = popart.AnchorReturnType("final")
SUM = popart.AnchorReturnType("sum")
EVERY1 = popart.AnchorReturnType("everyn", 1)
EVERY2 = popart.AnchorReturnType("everyn", 2)
EVERY4 = popart.AnchorReturnType("everyn", 4)


def generate_identity_inference_session_data():
    """Generates test cases for the test_identity_inference_session() test. See
    the test for the meaning of different arguments.

    Returns:
        A list of arguments to call the test with.
    """
    data = []

    # 0-d input tensors, batchesPerStep = 1
    data.append([[], 1, 1, ALL, 1, True, 1])
    data.append([[], 1, 1, ALL, 1, False, 1])
    data.append([[], 1, 1, EVERY1, 1, False, 1])
    data.append([[], 1, 1, FINAL, 1, False, 1])
    data.append([[], 1, 1, SUM, 1, True, 1])
    data.append([[], 1, 1, SUM, 1, False, 1])

    # 0-d input tensors, batchesPerStep > 1
    inputArray = [1, 2, 3, 4, 5, 6, 7, 8]
    data.append([[], inputArray, 8, ALL, 1, True, inputArray])
    data.append([[], inputArray, 8, ALL, 1, False, inputArray])
    data.append([[], inputArray, 8, EVERY4, 1, False, [4, 8]])
    data.append([[], inputArray, 8, FINAL, 1, False, 8])
    data.append([[], inputArray, 8, SUM, 1, True, 36])
    data.append([[], inputArray, 8, SUM, 1, False, 36])

    # 1-d input tensors, batchesPerStep = 1
    inputArray = [1, 2]
    data.append([[2], [1, 2], 1, ALL, 1, True, [1, 2]])
    data.append([[2], [1, 2], 1, ALL, 1, False, [1, 2]])
    data.append([[2], [1, 2], 1, EVERY1, 1, False, [1, 2]])
    data.append([[2], [1, 2], 1, FINAL, 1, False, [1, 2]])
    data.append([[2], [1, 2], 1, SUM, 1, True, [1, 2]])
    data.append([[2], [1, 2], 1, SUM, 1, False, [1, 2]])

    # 1-d input tensors, batchesPerStep > 1
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    data.append([[2], inputArray, 4, ALL, 1, True, inputArray])
    data.append([[2], inputArray, 4, ALL, 1, False, inputArray])
    data.append([[2], inputArray, 4, EVERY2, 1, False, [[3, 4], [7, 8]]])
    data.append([[2], inputArray, 4, FINAL, 1, False, [7, 8]])
    data.append([[2], inputArray, 4, SUM, 1, True, [16, 20]])
    data.append([[2], inputArray, 4, SUM, 1, False, [16, 20]])

    # 0-d input tensors, batchesPerStep > 1, replication = 2
    inputArray = [[1, 2], [3, 4], [5, 6], [7, 8]]
    data.append([[], inputArray, 4, ALL, 2, True, inputArray])
    data.append([[], inputArray, 4, ALL, 2, False, inputArray])
    data.append([[], inputArray, 4, EVERY2, 2, False, [[3, 4], [7, 8]]])
    data.append([[], inputArray, 4, FINAL, 2, False, [7, 8]])
    data.append([[], inputArray, 4, SUM, 2, True, [16, 20]])
    data.append([[], inputArray, 4, SUM, 2, False, [16, 20]])

    return data


@pytest.mark.parametrize("inputShape,inputArray,BPS,art,R,explicit,expected",
                         generate_identity_inference_session_data())
def test_identity_inference_session(inputShape, inputArray, BPS, art, R,
                                    explicit, expected):
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
    opts.enableExplicitMainLoops = explicit
    opts.useHostCopyOps = explicit

    device = tu.create_test_device(numIpus=R)
    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=device,
                                      userOptions=opts)

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {i1: np.array(inputArray, dtype=np.float32)}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    assert (np.array_equal(anchors[o], expected))


def generate_simple_training_session_data():
    """Generates test cases for the test_simple_training_session() test. See the
    test for the meaning of different arguments.

    Returns:
        A list of arguments to call the test with.
    """
    data = []

    # 1-d input tensors, batchesPerStep > 1, gradient accumulation = 2
    inputArray = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    data.append([[2], inputArray, 2, ALL, 2, True, inputArray])
    data.append([[2], inputArray, 2, ALL, 2, False, inputArray])
    data.append([[2], inputArray, 2, EVERY2, 2, False, [[3, 4], [7, 8]]])
    data.append([[2], inputArray, 2, FINAL, 2, False, [7, 8]])
    data.append([[2], inputArray, 2, SUM, 2, False, [16, 20]])

    # Test that art sum works with all permutations of bps = {1, 2}, ga = {1, 2}
    # and explicit main loops.
    inputArray = [[1, 2]]
    data.append([[2], inputArray, 1, SUM, 1, True, [1, 2]])
    inputArray = [[[1, 2], [3, 4]]]
    data.append([[2], inputArray, 1, SUM, 2, True, [4, 6]])
    inputArray = [[1, 2], [3, 4]]
    data.append([[2], inputArray, 2, SUM, 1, True, [4, 6]])
    inputArray = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    data.append([[2], inputArray, 2, SUM, 2, True, [16, 20]])

    return data


@pytest.mark.parametrize("inputShape,inputArray,BPS,art,GA,explicit,expected",
                         generate_simple_training_session_data())
def test_simple_training_session(inputShape, inputArray, BPS, art, GA,
                                 explicit, expected):
    builder = popart.Builder()

    inInfo = popart.TensorInfo("FLOAT", inputShape)

    i1 = builder.addInputTensor(inInfo)
    w1 = builder.addInitializedInputTensor(
        np.zeros(inputShape, dtype=np.float32), "w1")
    o = builder.aiOnnx.add([i1, w1])
    l1 = builder.aiGraphcore.l1loss([o], 0.0)

    proto = builder.getModelProto()

    batchesPerStep = BPS
    dataFlow = popart.DataFlow(batchesPerStep, {o: art})

    opts = popart.SessionOptions()
    opts.accumulationFactor = GA
    opts.enableGradientAccumulation = GA > 1
    opts.enableExplicitMainLoops = explicit
    opts.useHostCopyOps = explicit

    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=dataFlow,
                                     deviceInfo=tu.create_test_device(),
                                     userOptions=opts,
                                     loss=l1,
                                     optimizer=popart.ConstSGD(0.01))

    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()

    inputs = {i1: np.array(inputArray, dtype=np.float32)}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    assert (np.array_equal(anchors[o], expected))


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
