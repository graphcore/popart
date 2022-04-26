# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import popxl

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def is_hw(device: popart.DeviceInfo) -> bool:
    return device.type == popart.DeviceType.Ipu


def attach(device: popart.DeviceInfo) -> bool:
    # If OnDemand Hw, just calling `attach` will immediately return False,
    # not wait until it can attach.
    if is_hw(device):
        return device.tryAttachUntilTimeout()
    else:
        return device.attach()


def mk_session_with_test_device(ir: popxl.Ir) -> popxl.Session:
    session = popxl.Session(ir, 'cpu')
    session.device = tu.create_test_device(
        connectionType=popart.DeviceConnectionType.OnDemand).device
    return session
