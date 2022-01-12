# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import itertools

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_enum_specfic_devices():
    """Test that enumerating specific, per type and count works.
    """
    ipu_counts = [1, 2, 4, 8, 16]
    ipu_types = [popart.DeviceType.IpuModel, popart.DeviceType.Cpu]
    if tu.ipu_available():
        ipu_types += [popart.DeviceType.Ipu]
    deviceManager = popart.DeviceManager()
    for count, type_ in itertools.product(ipu_counts, ipu_types):
        devices = deviceManager.enumerateDevices(
            pattern=popart.SyncPattern.Full, numIpus=count, deviceType=type_)

        for device in devices:
            assert device.numIpus == count
            assert isinstance(device.type, type(type_))


@tu.requires_ipu
def test_aquire_device_by_id_0():
    """Test that aquiring by id works.
    """
    deviceManager = popart.DeviceManager()
    ipu_count = len(deviceManager.enumerateDevices(numIpus=1))
    for id_ in range(ipu_count):
        device = deviceManager.acquireDeviceById(
            id=id_, pattern=popart.SyncPattern.Full)
        # We can't be sure something else isn't using the IPUs...
        if device is not None:
            assert device.id == id_
            device.detach()


@tu.requires_ipu
def test_aquire_device_by_id_1():
    """
    Test that an error is thrown if trying to acquire device by id with
    an invalid id
    """
    deviceManager = popart.DeviceManager()
    invalidId = 999
    with pytest.raises(
            popart.poplar_exception,
            match="No such device set id: '" + str(invalidId) + "'") as e:
        deviceManager.acquireDeviceById(invalidId)


@tu.requires_ipu
def test_aquire_device_by_id_2():
    """
    Test that an error is thrown if trying to acquire device by id when it
    has already been attached to
    """
    deviceManager = popart.DeviceManager()
    deviceId = 0
    device = deviceManager.acquireDeviceById(deviceId)
    with pytest.raises(popart.popart_exception,
                       match="Failed to acquire device with id '0'") as e:
        deviceManager.acquireDeviceById(deviceId)


@tu.requires_ipu
def test_aquire_device_by_id_3():
    """
    Test that no error is thrown if trying to acquire device by id when it
    has already been attached to, and that a null device is returned.
    """
    deviceManager = popart.DeviceManager()
    deviceId = 0
    device0 = deviceManager.acquireDeviceById(deviceId)
    assert device0 != None
    device1 = deviceManager.tryAcquireDeviceById(deviceId)
    assert device1 == None


@tu.requires_ipu
def test_default_connection_type_0():
    deviceManager = popart.DeviceManager()
    device = deviceManager.acquireAvailableDevice(1)
    assert device.connectionType == popart.DeviceConnectionType.Always


@tu.requires_ipu
def test_default_connection_type_1():
    """
    Test that error is thrown if try to acquire more devices than are
    available
    """
    deviceManager = popart.DeviceManager()
    ipus = 256  # More IPUs than are available
    with pytest.raises(
            popart.popart_exception,
            match=
            "Failed to acquire device with 256 IPUs. Ensure that there are sufficient IPUs available"
    ) as e:
        deviceManager.acquireAvailableDevice(ipus)


@tu.requires_ipu
def test_default_connection_type_2():
    """
    Test that error is thrown if try to acquire a single IPU when all have
    already been attached to
    """
    deviceManager = popart.DeviceManager()
    availableIpus = len(deviceManager.enumerateDevices())
    d0 = deviceManager.acquireAvailableDevice(availableIpus)
    with pytest.raises(
            popart.popart_exception,
            match=
            f"Failed to acquire device with {1} IPUs. Ensure that there are sufficient IPUs available"
    ) as e:
        deviceManager.acquireAvailableDevice(1)


@tu.requires_ipu
def test_on_demand_connection_type():
    deviceManager = popart.DeviceManager()
    device = deviceManager.acquireAvailableDevice(
        1, connectionType=popart.DeviceConnectionType.OnDemand)
    assert device.connectionType == popart.DeviceConnectionType.OnDemand


@tu.requires_ipu
def test_attached_state():
    deviceManager = popart.DeviceManager()
    deviceManager.setOnDemandAttachTimeout(1)
    device = deviceManager.acquireAvailableDevice(
        1, connectionType=popart.DeviceConnectionType.OnDemand)
    assert not device.isAttached
    device.attach()
    assert device.isAttached
    device.detach()
    assert not device.isAttached
    device.tryAttachUntilTimeout()
    assert device.isAttached
    device.detach()
    assert not device.isAttached


@tu.requires_ipu_model
def test_set_and_get_ipu_model_version():
    dm = popart.DeviceManager()
    device = dm.createIpuModelDevice({'ipuVersion': 'ipu1'})
    assert device.version == "ipu1"

    device = dm.createIpuModelDevice({'ipuVersion': 'ipu2'})
    assert device.version == "ipu2"


@tu.requires_ipu_model
def test_check_default_ipu_model_0():
    dm = popart.DeviceManager()
    device = dm.createIpuModelDevice({})
    assert device.version == "ipu2"


@pytest.mark.parametrize("loadEngine", {True, False, None})
def test_prepareDevice_inference(loadEngine, capfd):
    device = tu.create_test_device()

    # Create a builder and construct a graph
    builder = popart.Builder()

    data_shape = [3]
    data_info = popart.TensorInfo("FLOAT", data_shape)

    a = builder.addInputTensor(data_info)
    b = builder.addInputTensor(data_info)

    o = builder.aiOnnx.add([a, b])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    # Describe how to run the model
    dataFlow = popart.DataFlow(2, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()

    # Create a session to compile and execute the graph
    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      userOptions=opts,
                                      deviceInfo=device)

    def assertLogContains(expectedCompilation, expectedEngineLoad):
        _, stderr = capfd.readouterr()
        graphCompiled = False
        engineLoaded = False
        for line in stderr.splitlines():
            if 'Graph compiled' in line:
                graphCompiled = True
            elif 'Engine loaded' in line:
                engineLoaded = True

        assert expectedCompilation == graphCompiled
        assert engineLoaded == expectedEngineLoad

    popart.getLogger().setLevel('INFO')
    with pytest.raises(popart.popart_exception,
                       match="no compiled engine") as e:
        session.loadEngineAndConnectStreams()

    # Compile graph
    if loadEngine is None:
        session.prepareDevice()
        # We expect the engine to be loaded by default
        loadEngine = True
    else:
        session.prepareDevice(loadEngine)
    assertLogContains(expectedCompilation=True, expectedEngineLoad=loadEngine)

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    # Generate some random input data
    data_shape.insert(0, 2)
    data_a = np.random.random_sample(data_shape).astype(np.float32)
    data_b = np.random.random_sample(data_shape).astype(np.float32)

    stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
    session.run(stepio)
    assertLogContains(expectedCompilation=False,
                      expectedEngineLoad=not loadEngine)

    assert np.allclose(anchors[o], data_a + data_b)


@pytest.mark.parametrize("loadEngine", {True, False, None})
def test_prepareDevice_training(loadEngine, capfd):
    filt_data = np.array([1., 2., 1., 2.], dtype=np.float32)
    filt_data = np.reshape(filt_data, [1, 1, 2, 2])
    input_data = np.array([1., 2., 3., 4.], dtype=np.float32)
    input_data = np.reshape(input_data, [1, 1, 2, 2])

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", input_data.shape)
    i1 = builder.addInputTensor(shape, "data")

    i2 = builder.addInitializedInputTensor(filt_data, "filter")

    c1 = builder.aiOnnx.conv([i1, i2],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[2, 2])

    l1 = builder.aiGraphcore.l1loss([c1],
                                    0.1,
                                    reduction=popart.ReductionType.Sum)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {c1: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableOutlining = False
    opts.enableOutliningCopyCostPruning = False

    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=dataFlow,
                                     userOptions=opts,
                                     optimizer=popart.ConstSGD(0.1),
                                     loss=l1,
                                     deviceInfo=tu.create_test_device())

    def assertLogContains(expectedCompilation, expectedEngineLoad):
        _, stderr = capfd.readouterr()
        graphCompiled = False
        engineLoaded = False
        for line in stderr.splitlines():
            if 'Graph compiled' in line:
                graphCompiled = True
            elif 'Engine loaded' in line:
                engineLoaded = True

        assert expectedCompilation == graphCompiled
        assert engineLoaded == expectedEngineLoad

    popart.getLogger().setLevel('INFO')
    # Compile graph
    if loadEngine is None:
        session.prepareDevice()
        # We expect the engine to be loaded by default
        loadEngine = True
    else:
        session.prepareDevice(loadEngine)
    assertLogContains(expectedCompilation=True, expectedEngineLoad=loadEngine)

    session.weightsFromHost()

    anchors = session.initAnchorArrays()

    inputs = {i1: input_data}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    assertLogContains(expectedCompilation=False,
                      expectedEngineLoad=not loadEngine)


def test_create_sim_device():
    dm = popart.DeviceManager()
    device = dm.createSimDevice({'numIPUs': 2, 'tilesPerIPU': 8})
    assert device.numIpus == 2
    assert device.tilesPerIPU == 8


def test_create_sim_default_device():
    dm = popart.DeviceManager()
    device = dm.createSimDevice({})
    assert device.numIpus == 1
    assert device.tilesPerIPU == 4
