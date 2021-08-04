# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu

import matmul_test_broadcasting_base as mtb

# Get really slow in engine compilation (backwards pass)
# " Engine compilation 82% complete "
#
shapes_ = (
    ([2, 1, 4, 5, 1, 7, 8], [2, 3, 1, 5, 6, 8, 9]),
    ([2, 4, 5, 1, 1, 8, 7], [2, 4, 5, 3, 6, 7, 9]),
)


def test_matmul_broadcasting_6(op_tester):
    mtb._test_matmul_broadcasting_base(op_tester, shapes_)
