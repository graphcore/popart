# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import popxl

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

import os


def get_test_device_with_timeout(numIpus: int):
    return tu.create_test_device(numIpus).device


def mk_session_with_test_device(ir: popxl.Ir) -> popxl.Session:
    test_target = os.environ.get('TEST_TARGET')
    if test_target == 'Cpu':
        dev = 'cpu'
    elif test_target == 'IpuModel':
        dev = 'ipu_model'
    elif test_target == 'Hw':
        dev = 'ipu_hw'
    else:
        raise ValueError('Unsupported TEST_TARGET =', test_target)

    return popxl.Session(ir, dev)
