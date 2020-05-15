# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu


def test_set_random_seed_error():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [10]))
    [o] = builder.aiOnnx.dropout([i1], num_outputs=1, ratio=0.3)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    s = popart.TrainingSession(fnModel=proto,
                               dataFlow=dataFlow,
                               optimizer=popart.ConstSGD(0.1),
                               losses=[popart.IdentityLoss(o, "idLossVal")],
                               userOptions=popart.SessionOptions(),
                               deviceInfo=tu.create_test_device(numIpus=2))

    with pytest.raises(popart.popart_exception) as e_info:
        s.setRandomSeed(0)

    msg = e_info.value.args[0]
    assert msg == ("Devicex::prepare() must be called before "
                   "Devicex::setRandomSeedFromHost(uint64_t) is called.")


def test_stochastic_rounding():
    # create test data
    d1 = np.random.rand(1, 3, 2, 2).astype(np.float16) * 100
    scale = np.random.rand(3).astype(np.float16)
    b = np.random.rand(3).astype(np.float16)
    mean = np.random.rand(3).astype(np.float16)
    var = np.random.rand(3).astype(np.float16)
    epsilon = 1e-05
    momentum = 0.1

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1))
    iScale = builder.addInitializedInputTensor(scale)
    iB = builder.addInitializedInputTensor(b)
    iMean = builder.addInitializedInputTensor(mean)
    iVar = builder.addInitializedInputTensor(var)
    [o_y, o_mean, o_var, o_smean, o_svar] = builder.aiOnnx.batchnormalization(
        [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)
    builder.addOutputTensor(o_y)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o_y: popart.AnchorReturnType("All")})

    device = tu.create_test_device()

    options = popart.SessionOptions()
    options.enableStochasticRounding = True

    sess = popart.TrainingSession(
        fnModel=proto,
        optimizer=popart.ConstSGD(0.1),
        losses=[popart.IdentityLoss(o_y, "idLossVal")],
        dataFlow=dataFlow,
        deviceInfo=device,
        userOptions=options)

    anchors = sess.initAnchorArrays()
    sess.prepareDevice()

    # Confirm that you are able to set the random seed when
    # enableStochasticRounding is true, even though the random seed tensor
    # is not consumed by any op in the Ir
    sess.setRandomSeed(0)


@tu.requires_ipu
def test_stochastic_rounding_behaviour():
    np.random.seed(seed=1)

    builder = popart.Builder()
    shape_d = [10]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT16", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT16", shape_d))
    out = builder.aiOnnx.add([d0, d1])

    opts = popart.SessionOptions()
    opts.enableStochasticRounding = True
    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
        userOptions=opts,
        deviceInfo=tu.create_test_device())

    anchors = session.initAnchorArrays()
    session.prepareDevice()

    data0 = np.ones(shape_d).astype(np.float16)
    data1 = 1e-3 * np.random.uniform(low=-1.0, high=1.0, size=shape_d).astype(
        np.float16)

    inputs = {d0: data0, d1: data1}
    stepio = popart.PyStepIO(inputs, anchors)

    print("Reference result:")
    reference = data0 + data1
    print("   ", reference)

    # Set the seed and run once to get 'expected result'
    session.setRandomSeed(1)
    session.run(stepio)

    # Observe that stochastic rounding has occured
    result0 = np.copy(anchors[out])
    assert np.array_equal(result0, reference) is False

    # Observe different stochastic rounding behaviour on the second run
    # after the seed is set
    session.run(stepio)
    result1 = np.copy(anchors[out])
    assert np.array_equal(result0, result1) is False

    # Observe different stochastic rounding behaviour on the first run
    # after a different seed is set
    session.setRandomSeed(12)
    session.run(stepio)
    assert np.array_equal(anchors[out], result0) is False

    print("Run:")
    for i in range(5):
        session.setRandomSeed(1)
        session.run(stepio)
        new_result0 = np.copy(anchors[out])
        session.run(stepio)
        new_result1 = np.copy(anchors[out])
        # Observe the same stochastic rounding behaviour across runs
        print(i, ":", new_result0)
        print("   ", new_result1)
        assert np.array_equal(new_result0, result0) is True
        assert np.array_equal(new_result1, result1) is True
