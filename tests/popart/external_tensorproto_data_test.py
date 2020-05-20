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
