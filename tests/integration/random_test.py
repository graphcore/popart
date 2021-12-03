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
    loss = builder.aiGraphcore.identityloss([o])

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    s = popart.TrainingSession(fnModel=proto,
                               dataFlow=dataFlow,
                               optimizer=popart.ConstSGD(0.1),
                               loss=loss,
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
    loss = builder.aiGraphcore.identityloss([o_y])
    builder.addOutputTensor(o_y)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o_y: popart.AnchorReturnType("All")})

    device = tu.create_test_device()

    options = popart.SessionOptions()
    options.enableStochasticRounding = True

    sess = popart.TrainingSession(fnModel=proto,
                                  optimizer=popart.ConstSGD(0.1),
                                  loss=loss,
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


@tu.requires_ipu
def test_reproducible_randomness_from_checkpoint(tmpdir):
    np.random.seed(seed=1)
    shape = [10, 10]
    d0_data = np.random.rand(*shape).astype(np.float32)
    in_data = np.random.rand(*shape).astype(np.float32)
    model_file_name = "model.onnx"

    builder = popart.Builder()
    t0 = builder.addInitializedInputTensor(d0_data)
    t1 = builder.addInputTensor("FLOAT", shape)
    t2 = builder.aiOnnx.dropout([t1], 1, 0.4)[0]
    mm = builder.aiOnnx.matmul([t0, t2])
    loss = builder.aiGraphcore.l1loss([mm], 0.1)

    def getSession(modelPath=""):
        if modelPath == "":
            fnProto = builder.getModelProto()
        else:
            fnProto = modelPath

        opts = popart.SessionOptions()
        opts.enableLoadAndOffloadRNGState = True

        session = popart.TrainingSession(userOptions=opts,
                                         fnModel=fnProto,
                                         dataFlow=popart.DataFlow(1, [mm]),
                                         loss=loss,
                                         optimizer=popart.ConstSGD(0.1),
                                         deviceInfo=tu.create_test_device())

        session.prepareDevice()
        session.weightsFromHost()
        return session

    # reference:
    #  - run once
    #  - save model after first session.run
    #  - run second time
    s0 = getSession()
    s0.prepareDevice()
    anchors0 = s0.initAnchorArrays()
    stepio0 = popart.PyStepIO({t1: in_data}, anchors0)
    s0.run(stepio0)
    s0.modelToHost(str(tmpdir / model_file_name))
    seed = s0.getRandomSeed()
    rngState = s0.getRNGState()  # Not really needed when SR is off.
    s0.run(stepio0)
    del (s0)  # free up IPU

    # from checkpoint:
    #  - load model from after s0's first session.run
    #  - run once
    s1 = getSession(modelPath=str(tmpdir / model_file_name))
    s1.prepareDevice()
    s1.setRandomSeed(seed)
    s1.setRNGState(rngState)  # Not really needed when SR is off.
    # NOTE: Order is important as setRandomSeed affects
    # RNG state.
    anchors1 = s1.initAnchorArrays()
    stepio1 = popart.PyStepIO({t1: in_data}, anchors1)
    s1.run(stepio1)

    # assert random behaviour is the same in the two
    # 'second session.runs'
    assert np.allclose(anchors0[mm], anchors1[mm])
