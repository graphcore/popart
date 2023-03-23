# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import torch
import popxl

from popxl.utils import (
    to_numpy,
    downcast_np_dtypes,
)
from popxl.fp8_utils import host_pow2scale_cast_to_fp8
from popxl.dtypes import _NP_TO_POPXL, _PT_TO_POPXL


# fmt: off
@pytest.mark.parametrize("x,dtype,downcast,exp_type", [
    # Numpy input
    [np.arange(4, dtype='float32'), None, True, np.dtype('float32')], # No downcast needed
    [np.arange(4, dtype='float64'), None, True, np.dtype('float32')], # Auto downcast float
    [np.arange(4, dtype='uint64'), None, True, np.dtype('uint32')], # Auto downcast uint
    [np.arange(4, dtype='float64'), None, False, np.dtype('float64')], # No downcast
    [np.arange(4, dtype='float64'), popxl.float32, False, np.dtype('float32')], # Manual downcast
    [np.arange(4, dtype='float32'), popxl.float64, True, np.dtype('float64')], # Manual upcast
    [np.arange(4, dtype='float32'), popxl.float8_143, False, popxl.dtypes.np_dtype_float8_143], # Float8 cast
    [np.arange(4, dtype='float64'), popxl.float8_152, False, popxl.dtypes.np_dtype_float8_152], # Float8 cast
    # Torch input
    [torch.arange(4, dtype=torch.float32), None, True, np.dtype('float32')], # No downcast needed
    [torch.arange(4, dtype=torch.float64), None, True, np.dtype('float32')], # Auto downcast float
    [torch.arange(4, dtype=torch.float64), None, False, np.dtype('float64')], # No downcast
    [torch.arange(4, dtype=torch.float64), popxl.float32, False, np.dtype('float32')], # Manual downcast
    [torch.arange(4, dtype=torch.float32), popxl.float64, True, np.dtype('float64')], # Manual upcast
    [torch.arange(4, dtype=torch.float32), popxl.float8_143, False, popxl.dtypes.np_dtype_float8_143], # Float8 cast
    [torch.arange(4, dtype=torch.float64), popxl.float8_152, False, popxl.dtypes.np_dtype_float8_152], # Float8 cast
    # List input
    [[0.0, 1.0, 2.0, 3.0], None, True, np.dtype('float32')], # Auto downcast float
    [[0, 1, 2, 3], None, True, np.dtype('int32')], # Auto downcast int
    [[0.0, 1.0, 2.0, 3.0], popxl.float32, False, np.dtype('float32')], # ?
    [[0.0, 1.0, 2.0, 3.0], popxl.float8_143, False, popxl.dtypes.np_dtype_float8_143], # Float8 cast
    [[0.0, 1.0, 2.0, 3.0], popxl.float8_152, False, popxl.dtypes.np_dtype_float8_152], # Float8 cast
    # Scalar input
    [0.0, None, True, np.dtype('float32')], # Auto downcast float
    [0, None, True, np.dtype('int32')], # Auto downcast int
    [False, None, True, np.dtype('bool')], # Bool input
    [0.1, popxl.float8_143, False, popxl.dtypes.np_dtype_float8_143], # Float8 cast
    [0.2, popxl.float8_152, False, popxl.dtypes.np_dtype_float8_152], # Float8 cast
    # Already converted float8 input.
    [host_pow2scale_cast_to_fp8(np.arange(4, dtype='float32'), popxl.float8_143), popxl.float8_143, False, popxl.dtypes.np_dtype_float8_143], # Float8 cast
])
def test_to_numpy(x, dtype, downcast, exp_type):
    """ Test to_numpy output types are correct. """
    out = to_numpy(x, dtype, downcast)
    assert isinstance(out, np.ndarray)
    assert exp_type == out.dtype
# fmt: on


@pytest.mark.parametrize("copy", (True, False))
@pytest.mark.parametrize(
    ("src", "dtype"),
    (
        *map(lambda d: ("pt", d), _PT_TO_POPXL.keys()),
        *map(lambda d: ("np", d), _NP_TO_POPXL.keys()),
    ),
)
def test_to_numpy_copy(src, dtype, copy):
    if src == "pt":
        t = torch.ones((2, 2), dtype=dtype)
    elif src == "np":
        t = np.ones((2, 2), dtype=dtype)

    result = to_numpy(t, copy=copy)
    if src == "pt":
        t_np = t.detach().numpy()
    elif src == "np":
        t_np = t

    should_alias = not copy and not (t_np.dtype in downcast_np_dtypes.keys())
    assert np.shares_memory(t_np, result) == should_alias


@pytest.mark.parametrize(("dtype", "result_dtype"), downcast_np_dtypes.items())
def test_to_numpy_downcast(dtype, result_dtype):
    t = np.ones((2, 2), dtype=dtype)
    assert to_numpy(t).dtype == result_dtype
