# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import os
import popart
import pytest
import tempfile


def test_save_tensors_externally():
    d1 = np.array([1, -1, 6]).astype(np.float32)
    d2 = np.array([7, 4]).astype(np.float16)
    builder = popart.Builder()
    i1 = builder.addInitializedInputTensor(d1)
    i2 = builder.addInitializedInputTensor(d2)
    o = builder.aiOnnx.add([i1, i2])
    tmpdir = tempfile.mkdtemp()

    def checkFile(file):
        # Check file exists
        assert os.path.exists(file)

        # Check file is of expected size: (3 * 4) + (2 * 2) = 16 bytes
        assert os.path.getsize(file) == 16

        # Read the binary data back in and check the value is as expected
        assert np.array_equal(np.fromfile(file, dtype=np.float32, count=3), d1)
        assert np.array_equal(
            np.fromfile(file, dtype=np.float16, count=2, offset=12), d2)

    # Test GraphTransformer
    tmpfile0 = os.path.join(tmpdir, "model_tensors0.onnx")
    graph_transformer = popart.GraphTransformer(builder.getModelProto())
    graph_transformer.saveInitializersExternally([i1, i2], tmpfile0)
    checkFile(tmpfile0)

    # Test Builder
    tmpfile1 = os.path.join(tmpdir, "model_tensors1.onnx")
    builder.saveInitializersExternally([i1, i2], tmpfile1)
    checkFile(tmpfile1)


def test_try_save_externally_when_already_external():
    builder = popart.Builder()
    i1 = builder.addInitializedInputTensor(np.ones([10], dtype=np.float32))
    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, "model_tensors.onnx")
    builder.saveInitializersExternally([i1], tmpfile)

    with pytest.raises(popart.popart_exception) as e_info:
        builder.saveInitializersExternally([i1], tmpfile)
    assert "already has an external data_location" in e_info.value.args[0]


def test_try_save_non_initializer_externally():
    builder = popart.Builder()
    c = builder.aiOnnx.constant(np.array([1, 6], dtype=np.float32))
    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, "model_tensors.onnx")

    with pytest.raises(popart.popart_exception) as e_info:
        builder.saveInitializersExternally([c], tmpfile)
    assert "is not an initializer" in e_info.value.args[0]


def test_load_externally_saved_tensors():
    """
    Test that initializer data can be saved in a separate file, and read into
    the PopART IR in an InferenceSession (by observing an expected inference
    result)
    """
    builder = popart.Builder()
    d1 = np.array([1, -1, 6]).astype(np.float32)
    d2 = np.array([-8, 7, 4]).astype(np.float32)
    i1 = builder.addInitializedInputTensor(d1)
    i2 = builder.addInitializedInputTensor(d2)
    o = builder.aiOnnx.add([i1, i2])
    tmpdir = tempfile.mkdtemp()
    tmpfile_tensors = os.path.join(tmpdir, "tensors.onnx")
    tmpfile_model = os.path.join(tmpdir, "model.onnx")
    builder.saveInitializersExternally([i1, i2], tmpfile_tensors)
    builder.saveModelProto(tmpfile_model)

    # Create builder from onnx model
    builder = popart.Builder(tmpfile_model)
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=dataFlow,
        deviceInfo=popart.DeviceManager().createCpuDevice())
    anchors = session.initAnchorArrays()
    session.prepareDevice()
    stepio = popart.PyStepIO({}, anchors)
    session.run(stepio)
    assert (np.array_equal(anchors[o], d1 + d2))


def test_save_back_externally_saved_tensors():
    """
    Test that initializers (stored externally in the onnx model) that are
    updated in a training session are written back correctly when the onnx
    model is written using the Session API
    Model:
    in0 -
          \
           Matmul0 - Matmul1 - out
          /          /
    w0 --       w1--
    """
    builder = popart.Builder()
    shape = [4, 4]
    elms = np.prod(shape)
    numLayers = 2
    in0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape))
    initWeights = []
    weightsIds = []
    anchorsDef = {}
    out = in0
    for layer in range(numLayers):
        w_init = np.random.rand(*shape).astype('float32')
        initWeights.append(w_init)
        weightsIds.append(builder.addInitializedInputTensor(w_init))
        anchorsDef[weightsIds[layer]] = popart.AnchorReturnType("All")
        out = builder.aiOnnx.matmul([out, weightsIds[layer]])

    loss = builder.aiGraphcore.identityloss([out])
    tmpdir = tempfile.mkdtemp()
    tmpfile_weights = os.path.join(tmpdir, "weights.onnx")
    builder.saveInitializersExternally(weightsIds, tmpfile_weights)

    # Verify the initial weights are saved correctly
    for layer in range(numLayers):
        saved_weights = np.fromfile(tmpfile_weights,
                                    dtype=np.float32,
                                    count=elms,
                                    offset=layer * elms * 4)
        assert (np.array_equal(initWeights[layer].flatten(), saved_weights))

    opts = popart.SessionOptions()
    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(1, anchorsDef),
        deviceInfo=popart.DeviceManager().createCpuDevice(),
        optimizer=popart.ConstSGD(10),
        loss=loss)

    anchors = session.initAnchorArrays()
    inputs = {in0: np.random.rand(*shape).astype('float32')}
    stepio = popart.PyStepIO(inputs, anchors)

    session.prepareDevice()
    session.weightsFromHost()

    session.run(stepio)

    # Check the weights have been updated
    for layer in range(numLayers):
        assert not np.allclose(anchors[weightsIds[layer]], initWeights[layer])

    # Save the model with updated weights back to disk
    tmpfile_model = os.path.join(tmpdir, "model.onnx")
    session.modelToHost(tmpfile_model)

    # Verify that the file containing tensor data has also been updated
    for layer in range(numLayers):
        saved_weights = np.fromfile(tmpfile_weights,
                                    dtype=np.float32,
                                    count=elms,
                                    offset=layer * elms * 4)
        assert np.array_equal(anchors[weightsIds[layer]].flatten(),
                              saved_weights)


optimizerInfos = []
# 1. SGD with momentum
optimizerInfos.append((popart.SGD({
    "defaultLearningRate": (0.2, True),
    "defaultMomentum": (0.5, True)
}), [popart.reservedAcclPrefix()]))
# 2. Adam
optimizerInfos.append((popart.Adam({
    "defaultLearningRate": (0.2, True),
    "defaultBeta1": (0.1, True),
    "defaultBeta2": (0.1, True),
    "defaultWeightDecay": (0.5, True),
    "defaultEps": (1e-5, True),
    "lossScaling": (2, True)
}), [
    popart.reservedAccl1Prefix(),
    popart.reservedAccl2Prefix(),
    popart.reservedStepPrefix()
]))
# 3. Adaptive
optimizerInfos.append(
    (popart.Adaptive({"defaultLearningRate": (0.2, True)},
                     mode=popart.AdaptiveMode.CenteredRMSProp), [
                         popart.reservedAccl1Prefix(),
                         popart.reservedAccl2Prefix()
                     ]))


@pytest.mark.parametrize("optimizerInfo", optimizerInfos)
def test_save_tensors_optimizer_state_externally(optimizerInfo):
    """
    # 1. create training session with momentum, save initializers externally
    # 2. check file size before session.modelToHost, see it grows after
    #    due to the additional optimizer state tensors being saved
    # 3. read tensors from file, compare with anchors to verify that the
    #    additional optimizer state has saved correctly
    # 4. Create a new session from the saved onnx model, run both sessions
    #    and compare outputs to verify that the optimizer state tensors
    #    were loaded in correctly
    """
    optimizer = optimizerInfo[0]
    extraOptimizerStatePrefs = optimizerInfo[1]

    d1 = np.random.rand(3, 3).astype(np.float32)
    d2 = np.random.rand(3).astype(np.float32)
    builder = popart.Builder()
    i1 = builder.addInitializedInputTensor(d1)
    i2 = builder.addInitializedInputTensor(d2)
    o = builder.aiOnnx.matmul([i1, i2])
    loss = builder.aiGraphcore.identityloss([o])

    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, "model_tensors.onnx")
    builder.saveInitializersExternally([i1, i2], tmpfile)

    # Check file is of expected size: (3 * 3 * 4) + (3 * 4) = 48
    assert os.path.exists(tmpfile)
    assert os.path.getsize(tmpfile) == d1.size * 4 + d2.size * 4

    anchorIds = [o]
    anchorIds.append(popart.reservedGradientPrefix() + i1)
    anchorIds.append(popart.reservedGradientPrefix() + i2)

    session = popart.TrainingSession(
        deviceInfo=popart.DeviceManager().createCpuDevice(),
        fnModel=builder.getModelProto(),
        loss=loss,
        optimizer=optimizer,
        dataFlow=popart.DataFlow(1, anchorIds))

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    session.run(popart.PyStepIO({}, anchors))

    session.weightsToHost()
    weightsMap = {}
    weightsMap[i1] = np.ones(d1.size).astype(np.float32)
    weightsMap[i2] = np.ones(d2.size).astype(np.float32)
    for pref in extraOptimizerStatePrefs:
        if pref == popart.reservedStepPrefix():
            size1 = 1
            size2 = 1
        else:
            size1 = d1.size
            size2 = d2.size
        weightsMap[pref + i1] = np.ones(size1).astype(np.float32)
        weightsMap[pref + i2] = np.ones(size2).astype(np.float32)
    session.readWeights(popart.PyWeightsIO(weightsMap))

    tmpfile1 = os.path.join(tmpdir, "model.onnx")
    session.modelToHost(tmpfile1)

    # Extra state for each initializer
    expectedSize = (d1.size * 4) + (d2.size * 4)
    for pref in extraOptimizerStatePrefs:
        if pref == popart.reservedStepPrefix():
            expectedSize += (2 * 4)
        else:
            expectedSize += d1.size * 4
            expectedSize += d2.size * 4

    assert os.path.getsize(tmpfile) == expectedSize

    # Compare anchors with external data written to file
    saved_weights = np.fromfile(tmpfile, dtype=np.float32)
    assert np.allclose(saved_weights[0:d1.size], weightsMap[i1].flatten())
    totalSize = d1.size + d2.size
    assert np.allclose(saved_weights[d1.size:totalSize],
                       weightsMap[i2].flatten())

    for pref in extraOptimizerStatePrefs:
        assert np.allclose(saved_weights[totalSize:totalSize + d1.size],
                           weightsMap[pref + i1].flatten())
        totalSize += d1.size
        assert np.allclose(saved_weights[totalSize:totalSize + d2.size],
                           weightsMap[pref + i2].flatten())
        totalSize += d2.size

    # Create new session
    new_session = popart.TrainingSession(
        deviceInfo=popart.DeviceManager().createCpuDevice(),
        fnModel=tmpfile1,
        loss=loss,
        optimizer=optimizer,
        dataFlow=popart.DataFlow(1, anchorIds))
    new_anchors = new_session.initAnchorArrays()
    new_session.prepareDevice()
    new_session.weightsFromHost()

    new_session.run(popart.PyStepIO({}, new_anchors))
    session.run(popart.PyStepIO({}, anchors))

    # Compare output from both sessions to confirm that the optimizer state
    # tensors have been read back in correctly for the new session
    for anchorId in anchorIds:
        assert np.allclose(anchors[anchorId], new_anchors[anchorId])
