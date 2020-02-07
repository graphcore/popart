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
    if tu.ipu_avaliable:
        ipu_types += [popart.DeviceType.Ipu]
    deviceManager = popart.DeviceManager()
    for count, type_ in list(itertools.product(ipu_counts, ipu_types)):
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
    for id_ in list(range(16)):
        device = deviceManager.acquireDeviceById(
            id=id_, pattern=popart.SyncPattern.Full)
        # We can't be sure something else isn't using the IPUs...
        if device is not None:
            assert device.id == id_
            device.detach()
