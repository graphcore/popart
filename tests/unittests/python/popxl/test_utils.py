# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
from popxl.utils import to_numpy
import popxl
import torch
import numpy as np


# fmt: off
@pytest.mark.parametrize("x,dtype,downcast,expected_type", [
    # Numpy input
    [np.arange(4, dtype='float32'), None, True, np.dtype('float32')], # No downcast needed
    [np.arange(4, dtype='float64'), None, True, np.dtype('float32')], # Auto downcast float
    [np.arange(4, dtype='uint64'), None, True, np.dtype('uint32')], # Auto downcast uint
    [np.arange(4, dtype='float64'), None, False, np.dtype('float64')], # No downcast
    [np.arange(4, dtype='float64'), popxl.float32, False, np.dtype('float32')], # Manual downcast
    [np.arange(4, dtype='float32'), popxl.float64, True, np.dtype('float64')], # Manual upcast
    # Torch input
    [torch.arange(4, dtype=torch.float32), None, True, np.dtype('float32')], # No downcast needed
    [torch.arange(4, dtype=torch.float64), None, True, np.dtype('float32')], # Auto downcast float
    [torch.arange(4, dtype=torch.float64), None, False, np.dtype('float64')], # No downcast
    [torch.arange(4, dtype=torch.float64), popxl.float32, False, np.dtype('float32')], # Manual downcast
    [torch.arange(4, dtype=torch.float32), popxl.float64, True, np.dtype('float64')], # Manual upcast
    # List input
    [[0.0, 1.0, 2.0, 3.0], None, True, np.dtype('float32')], # Auto downcast float
    [[0, 1, 2, 3], None, True, np.dtype('int32')], # Auto downcast int
    [[0.0, 1.0, 2.0, 3.0], popxl.float32, False, np.dtype('float32')], # Manual downcast
    # Scalar input
    [0.0, None, True, np.dtype('float32')], # Auto downcast float
    [0, None, True, np.dtype('int32')], # Auto downcast int
    [False, None, True, np.dtype('bool')], # Bool input
])
def test_to_numpy(x, dtype, downcast, expected_type):

    out = to_numpy(x, dtype, downcast)
    assert isinstance(out, np.ndarray)
    assert expected_type == out.dtype

# fmt: on
