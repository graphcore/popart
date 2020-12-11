# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
def test_aquire_device_by_id():
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
def test_default_connection_type():
    deviceManager = popart.DeviceManager()
    device = deviceManager.acquireAvailableDevice(1)
    assert device.connectionType == popart.DeviceConnectionType.Always


@tu.requires_ipu
def test_on_demand_connection_type():
    deviceManager = popart.DeviceManager()
    device = deviceManager.acquireAvailableDevice(
        1, connectionType=popart.DeviceConnectionType.OnDemand)
    assert device.connectionType == popart.DeviceConnectionType.OnDemand


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
