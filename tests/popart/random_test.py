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

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    s = popart.TrainingSession(fnModel=proto,
                               dataFeed=dataFlow,
                               optimizer=popart.ConstSGD(0.1),
                               losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
                               userOptions=popart.SessionOptionsCore(),
                               deviceInfo=tu.get_ipu_model(numIPUs=2))

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
    [o_y, o_mean, o_var, o_smean, o_svar] = builder.aiOnnx.batchnormalization([i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)
    builder.addOutputTensor(o_y)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o_y: popart.AnchorReturnType("ALL")})

    device = popart.DeviceManager().createCpuDevice()

    options = popart.SessionOptionsCore()
    options.enableStochasticRounding = True

    sess = popart.TrainingSession(fnModel=proto,
                                  optimizer=popart.ConstSGD(0.1),
                                  losses=[popart.L1Loss(o_y, "l1LossVal", 0.1)],
                                  dataFeed=dataFlow,
                                  deviceInfo=device,
                                  userOptions=options)

    anchors = sess.initAnchorArrays()
    sess.prepareDevice()

    # Confirm that you are able to set the random seed when
    # enableStochasticRounding is true, even though the random seed tensor
    # is not consumed by any op in the Ir
    sess.setRandomSeed(0)
