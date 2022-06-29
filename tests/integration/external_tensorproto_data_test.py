# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import os
import popart
import pytest
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from pathlib import Path


# Context manager changes directory and changes back when context exits.
@contextmanager
def change_directory(path):
    if not isinstance(path, Path):
        path = Path(path)

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def test_save_tensors_externally():
    d1 = np.array([1]).astype(np.float32)
    d2 = np.array([7, 4]).astype(np.float16)
    builder = popart.Builder()
    i1 = builder.addInitializedInputTensor(d1)
    i2 = builder.addInitializedInputTensor(d2)
    _ = builder.aiOnnx.add([i1, i2])
    with TemporaryDirectory() as tmpdir:

        def checkFile(file):
            # Check file exists
            assert os.path.exists(file)

            # Check file is of expected size: (1 * 4) + (2 * 2) = 16 bytes
            assert os.path.getsize(file) == 8

            # Read the binary data back in and check the value is as expected
            assert np.array_equal(np.fromfile(file, dtype=np.float32, count=1),
                                  d1)
            assert np.array_equal(
                np.fromfile(file, dtype=np.float16, count=2, offset=4), d2)

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
    with TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "model_tensors.onnx")
        builder.saveInitializersExternally([i1], tmpfile)

        with pytest.raises(popart.popart_exception) as e_info:
            builder.saveInitializersExternally([i1], tmpfile)
        assert "already has an external data_location" in e_info.value.args[0]


def test_try_save_non_initializer_externally():
    builder = popart.Builder()
    c = builder.aiOnnx.constant(np.array([1, 6], dtype=np.float32))
    with TemporaryDirectory() as tmpdir:
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
    with TemporaryDirectory() as tmpdir:
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
    with TemporaryDirectory() as tmpdir:
        tmpfile_weights = os.path.join(tmpdir, "weights.onnx")
        builder.saveInitializersExternally(weightsIds, tmpfile_weights)

        # Verify the initial weights are saved correctly
        for layer in range(numLayers):
            saved_weights = np.fromfile(tmpfile_weights,
                                        dtype=np.float32,
                                        count=elms,
                                        offset=layer * elms * 4)
            assert (np.array_equal(initWeights[layer].flatten(),
                                   saved_weights))

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
            assert not np.allclose(anchors[weightsIds[layer]],
                                   initWeights[layer])

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

    with TemporaryDirectory() as tmpdir:
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


def test_external_location_relative_path():
    # Demonstrate that external data can be saved using a path relative
    # to the cwd
    builder = popart.Builder()
    i1 = builder.addInitializedInputTensor(np.array([7, 4]).astype(np.float16))
    _ = builder.aiOnnx.add([i1, i1])

    with TemporaryDirectory() as tmpdir:
        tensor_data_file = "model_tensors.onnx"
        tmpfile = os.path.join(tmpdir, tensor_data_file)
        with change_directory(tmpdir):
            # Save tensor data externally, using a relative path
            builder.saveInitializersExternally([i1], tensor_data_file)
            assert os.path.exists(tmpfile)
            builder.saveModelProto("model.onnx")

            # Can load tensor data from cwd, because it relative path exists
            _ = popart.Builder("model.onnx")

            # Fail to load tensor data, because relative path does not exist in this
            # case
            os.mkdir("dummy_dir")
            with change_directory("dummy_dir"):
                with pytest.raises(RuntimeError) as e_info:
                    _ = popart.Builder("../model.onnx")
                assert "model_tensors.onnx, but it doesn't exist or is not accessible" in e_info.value.args[
                    0]


def test_external_location_path_does_not_exist():
    # Try to save tensors externally to file whose parent directory doesn't
    # exist
    builder = popart.Builder()
    i1 = builder.addInitializedInputTensor(
        np.array([-1, 6]).astype(np.float32))
    _ = builder.aiOnnx.add([i1, i1])

    with TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "dummy/tensors.onnx")

        with pytest.raises(popart.popart_exception) as e_info:
            builder.saveInitializersExternally([i1], tmpfile)
        assert "parent directory does not exist" in e_info.value.args[0]


def test_overwriting_external_data_file():
    # Verify that if calling modelToHost twice, the external data is overwritten
    # correctly, and not corrupted!
    builder = popart.Builder()
    d1 = np.random.rand(3, 3).astype(np.float32)
    i1 = builder.addInitializedInputTensor(d1)
    o = builder.aiOnnx.matmul([i1, i1])
    loss = builder.aiGraphcore.identityloss([o])

    with TemporaryDirectory() as tmpdir:
        tmpfile0 = os.path.join(tmpdir, "model_tensors0.onnx")
        builder.saveInitializersExternally([i1], tmpfile0)

        optimizer = popart.SGD({
            "defaultLearningRate": (0.2, True),
            "defaultMomentum": (0.5, True)
        })

        session = popart.TrainingSession(
            deviceInfo=popart.DeviceManager().createCpuDevice(),
            fnModel=builder.getModelProto(),
            loss=loss,
            optimizer=optimizer,
            dataFlow=popart.DataFlow(1, []))

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        session.run(popart.PyStepIO({}, anchors))

        # Should overwrite external data with the same data
        tmpfile1 = os.path.join(tmpdir, "model0.onnx")
        session.modelToHost(tmpfile1)
        weights0 = np.fromfile(tmpfile0, dtype=np.float32)
        session.modelToHost(tmpfile1)
        weights1 = np.fromfile(tmpfile0, dtype=np.float32)
        assert np.array_equal(weights0, weights1)


def test_checkpointing_with_externally_stored_tensor_data0():
    # Test demonstrating how checkpointing of model weights when using
    # externally saved tensor data does not work by using a relative path
    # in the external tensor info.
    #   - New external data files are not created unless there is an explicit
    #     call
    builder = popart.Builder()
    i1 = builder.addInitializedInputTensor(
        np.array([1, -1]).astype(np.float32))
    o = builder.aiOnnx.add([i1, i1])

    with TemporaryDirectory() as tmpdir:
        with change_directory(tmpdir):
            # Setting the (relative) path of tensor data for the first and only time
            builder.saveInitializersExternally([i1], "tensors.onnx")

            # Create builder from onnx model
            opts = popart.SessionOptions()
            opts.constantWeights = False
            session = popart.InferenceSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(1, [o]),
                deviceInfo=popart.DeviceManager().createCpuDevice())
            anchors = session.initAnchorArrays()
            session.prepareDevice()

            # Run once, try to checkpoint.
            # Fails because session.modelToHost should not be able to write external
            # tensor data to a new location unless explicitly asked to (see
            # Session::updateExternallySavedTensorLocations)
            session.run(popart.PyStepIO({}, anchors))
            os.mkdir("checlpoint0")
            with change_directory("checlpoint0"):
                with pytest.raises(popart.popart_exception) as e_info:
                    session.modelToHost("model.onnx")
                assert "Unrecognised file name 'tensors.onnx" in e_info.value.args[
                    0]

                # New external data file has not been created
                assert not os.path.exists("tensors.onnx")


def test_checkpointing_with_externally_stored_tensor_data1():
    # Test demonstrating checkpointing of model weights when using externally
    # saved tensor data
    builder = popart.Builder()
    d1 = np.random.rand(3, 3).astype(np.float32)
    i1 = builder.addInitializedInputTensor(d1)
    o = builder.aiOnnx.matmul([i1, i1])
    loss = builder.aiGraphcore.identityloss([o])

    with TemporaryDirectory() as tmpdir:
        tmpfile0 = os.path.join(tmpdir, "model_tensors0.onnx")
        builder.saveInitializersExternally([i1], tmpfile0)

        optimizer = popart.SGD({
            "defaultLearningRate": (0.2, True),
            "defaultMomentum": (0.5, True)
        })

        session = popart.TrainingSession(
            deviceInfo=popart.DeviceManager().createCpuDevice(),
            fnModel=builder.getModelProto(),
            loss=loss,
            optimizer=optimizer,
            dataFlow=popart.DataFlow(1, []))

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        session.run(popart.PyStepIO({}, anchors))

        # Get baseline external weights from disk after one run
        tmpfile1 = os.path.join(tmpdir, "model0.onnx")
        session.modelToHost(tmpfile1)
        weights0 = np.fromfile(tmpfile0, dtype=np.float32)

        # Calling modelToHost without updating external locations - overwrites
        # existing external data in tmpfile0
        session.run(popart.PyStepIO({}, anchors))
        session.modelToHost(tmpfile1)
        weights1 = np.fromfile(tmpfile0, dtype=np.float32)
        assert not np.array_equal(weights0, weights1)

        # Update external weight location.
        # Save the onnx model to a new location, without running the session.
        # Confirm the tensor data in new weights file is the same as previously
        tmpfile2 = os.path.join(tmpdir, "model_tensors1.onnx")
        session.updateExternallySavedTensorLocations(tmpfile0, tmpfile2)
        assert os.path.exists(tmpfile2)
        tmpfile3 = os.path.join(tmpdir, "model1.onnx")
        session.modelToHost(tmpfile3)
        assert np.array_equal(np.fromfile(tmpfile2, dtype=np.float32),
                              weights1)

        # Update external weight location.
        # Save the onnx model to a new location, this time with running the session.
        # Confirm the tensor data in new weights file has changed
        session.run(popart.PyStepIO({}, anchors))
        tmpfile4 = os.path.join(tmpdir, "model_tensors2.onnx")
        session.updateExternallySavedTensorLocations(tmpfile2, tmpfile4)
        assert os.path.exists(tmpfile4)
        tmpfile5 = os.path.join(tmpdir, "model2.onnx")
        session.modelToHost(tmpfile5)
        assert not np.array_equal(np.fromfile(tmpfile4, dtype=np.float32),
                                  weights1)


def test_invalid_tensor_location_updates():
    # Test to demonstrate exceptions thrown during incorrect usage of
    # Session::updateExternallySavedTensorLocations
    builder = popart.Builder()
    d1 = np.random.rand(3, 3).astype(np.float32)
    i1 = builder.addInitializedInputTensor(d1)
    o = builder.aiOnnx.matmul([i1, i1])
    loss = builder.aiGraphcore.identityloss([o])

    with TemporaryDirectory() as tmpdir:
        origpath = os.path.join(tmpdir, "model_tensors0.onnx")
        builder.saveInitializersExternally([i1], origpath)

        optimizer = popart.SGD({
            "defaultLearningRate": (0.2, True),
            "defaultMomentum": (0.5, True)
        })

        session = popart.TrainingSession(
            deviceInfo=popart.DeviceManager().createCpuDevice(),
            fnModel=builder.getModelProto(),
            loss=loss,
            optimizer=optimizer,
            dataFlow=popart.DataFlow(1, []))

        updatedpath0 = os.path.join(tmpdir, "model_tensors1.onnx")

        # Try to update from from a path that doesn't exist
        fakepath = os.path.join(tmpdir, "foo.bar")
        with pytest.raises(popart.popart_exception) as e_info:
            session.updateExternallySavedTensorLocations(
                fakepath, updatedpath0)
        assert "but file '" + fakepath + "' does not exist" in e_info.value.args[
            0]

        session.updateExternallySavedTensorLocations(origpath, updatedpath0)

        # Try to update from from old path
        updatedpath1 = os.path.join(tmpdir, "model_tensors2.onnx")
        with pytest.raises(popart.popart_exception) as e_info:
            session.updateExternallySavedTensorLocations(
                origpath, updatedpath1)
        assert "No ONNX model initializers have external location set to" in e_info.value.args[
            0]
