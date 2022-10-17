# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import torch
import popxl

from popxl.utils import (
    to_numpy,
    downcast_np_dtypes,
    host_pow2scale_then_cast,
    host_cast_then_pow2scale,
)
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
    [np.arange(4, dtype='float32'), popxl.float8_143, False, popxl.utils.np_dtype_float8_143], # Float8 cast
    [np.arange(4, dtype='float64'), popxl.float8_152, False, popxl.utils.np_dtype_float8_152], # Float8 cast
    # Torch input
    [torch.arange(4, dtype=torch.float32), None, True, np.dtype('float32')], # No downcast needed
    [torch.arange(4, dtype=torch.float64), None, True, np.dtype('float32')], # Auto downcast float
    [torch.arange(4, dtype=torch.float64), None, False, np.dtype('float64')], # No downcast
    [torch.arange(4, dtype=torch.float64), popxl.float32, False, np.dtype('float32')], # Manual downcast
    [torch.arange(4, dtype=torch.float32), popxl.float64, True, np.dtype('float64')], # Manual upcast
    [torch.arange(4, dtype=torch.float32), popxl.float8_143, False, popxl.utils.np_dtype_float8_143], # Float8 cast
    [torch.arange(4, dtype=torch.float64), popxl.float8_152, False, popxl.utils.np_dtype_float8_152], # Float8 cast
    # List input
    [[0.0, 1.0, 2.0, 3.0], None, True, np.dtype('float32')], # Auto downcast float
    [[0, 1, 2, 3], None, True, np.dtype('int32')], # Auto downcast int
    [[0.0, 1.0, 2.0, 3.0], popxl.float32, False, np.dtype('float32')], # ?
    [[0.0, 1.0, 2.0, 3.0], popxl.float8_143, False, popxl.utils.np_dtype_float8_143], # Float8 cast
    [[0.0, 1.0, 2.0, 3.0], popxl.float8_152, False, popxl.utils.np_dtype_float8_152], # Float8 cast
    # Scalar input
    [0.0, None, True, np.dtype('float32')], # Auto downcast float
    [0, None, True, np.dtype('int32')], # Auto downcast int
    [False, None, True, np.dtype('bool')], # Bool input
    [0.1, popxl.float8_143, False, popxl.utils.np_dtype_float8_143], # Float8 cast
    [0.2, popxl.float8_152, False, popxl.utils.np_dtype_float8_152], # Float8 cast
    # Already converted float8 input.
    [host_pow2scale_then_cast(np.arange(4, dtype='float32'), popxl.float8_143), popxl.float8_143, False, popxl.utils.np_dtype_float8_143], # Float8 cast
])
def test_to_numpy(x, dtype, downcast, exp_type):
    """ Test to_numpy output types are correct. """
    out = to_numpy(x, dtype, downcast)
    assert isinstance(out, np.ndarray)
    assert exp_type == out.dtype
# fmt: on

# fmt: off
@pytest.mark.parametrize("src,dtype,log2_scale,nan_on_overflow,exp_vals", [
    # Example data points:
    #
    # Legend:
    #
    #   S    = sign
    #   EEEE = exponent
    #   MMM  = mantissa
    #
    # How to interpret float8_143:
    #
    #                      S    MMM        EEEE format_bias                     log2_scale
    #                      |     |          |      |                                     |
    #   SEEEEMMM           v     v          v      v                                     v
    # 0b00111100           + 0b1.100 * 2^(0b0111 - 8) = 0b0.1100   = 0.75    = 0.75 * 2^ 0
    # 0b00101100           + 0b1.100 * 2^(0b0101 - 8) = 0b0.001100 = 0.1875  = 0.75 * 2^-2
    # 0b01000100           + 0b1.100 * 2^(0b1000 - 8) = 0b1.100    = 1.5     = 0.75 * 2^ 1
    #
    # How to interpret float8_152:
    #
    #                      S     MM       EEEEE format_bias                     log2_scale
    #                      |     |         |       |                                     |
    #   SEEEEEMM           v     v         v       v                                     v
    # 0b00111000           + 0b1.00 * 2^(0b01110 - 16) = 0b0.01    = 0.25    = 0.25 * 2^ 0
    # 0b00101100           + 0b1.00 * 2^(0b01011 - 16) = 0b0.00001 = 0.03125 = 0.25 * 2^-3
    #
    # src                        , dtype           ,l2s, nanoo, exp_vals
    # ===                        , =====           ,===, =====, ========
    # Test log2_scale for float8_143 + float32.
    [np.array([0.75], np.float32), popxl.float8_143,  0, False, np.array([0b00111100], popxl.utils.np_dtype_float8_143)],
    [np.array([0.75], np.float32), popxl.float8_143, -2, False, np.array([0b00101100], popxl.utils.np_dtype_float8_143)],
    [np.array([0.75], np.float32), popxl.float8_143,  1, False, np.array([0b01000100], popxl.utils.np_dtype_float8_143)],
    # Test log2_scale for float8_152 + float32.
    [np.array([0.25], np.float32), popxl.float8_152,  0, False, np.array([0b00111000], popxl.utils.np_dtype_float8_152)],
    [np.array([0.25], np.float32), popxl.float8_152, -3, False, np.array([0b00101100], popxl.utils.np_dtype_float8_152)],
    # Test log2_scale for float8_143 + float64.
    [np.array([0.75], np.float64), popxl.float8_143,  0, False, np.array([0b00111100], popxl.utils.np_dtype_float8_143)],
    [np.array([0.75], np.float64), popxl.float8_143, -2, False, np.array([0b00101100], popxl.utils.np_dtype_float8_143)],
    # Test log2_scale for float8_152 + float64.
    [np.array([0.75], np.float64), popxl.float8_143,  1, False, np.array([0b01000100], popxl.utils.np_dtype_float8_143)],
    [np.array([0.25], np.float64), popxl.float8_152,  0, False, np.array([0b00111000], popxl.utils.np_dtype_float8_152)],
    [np.array([0.25], np.float64), popxl.float8_152, -3, False, np.array([0b00101100], popxl.utils.np_dtype_float8_152)],
    # Test nan_on_overflow for float8_143 + float32.
    [np.array([250.], np.float32), popxl.float8_143,  0, False, np.array([0b01111111], popxl.utils.np_dtype_float8_143)],
    [np.array([250.], np.float32), popxl.float8_143,  0, True,  np.array([0b10000000], popxl.utils.np_dtype_float8_143)],
    # Test nan_on_overflow for float8_152 + float32.
    [np.array([70000.], np.float32), popxl.float8_152,  0, False, np.array([0b01111111], popxl.utils.np_dtype_float8_152)],
    [np.array([70000.], np.float32), popxl.float8_152,  0, True,  np.array([0b10000000], popxl.utils.np_dtype_float8_152)],
    # Test nan_on_overflow for float8_143 + float64.
    [np.array([250.], np.float64), popxl.float8_143,  0, False, np.array([0b01111111], popxl.utils.np_dtype_float8_143)],
    [np.array([250.], np.float64), popxl.float8_143,  0, True,  np.array([0b10000000], popxl.utils.np_dtype_float8_143)],
    # Test nan_on_overflow for float8_152 + float64.
    [np.array([70000.], np.float64), popxl.float8_152,  0, False, np.array([0b01111111], popxl.utils.np_dtype_float8_152)],
    [np.array([70000.], np.float64), popxl.float8_152,  0, True,  np.array([0b10000000], popxl.utils.np_dtype_float8_152)],
    # Test it works with torch inputs.
    [torch.tensor([0.75], dtype=torch.float32), popxl.float8_143,  0, False, np.array([0b00111100], popxl.utils.np_dtype_float8_143)],
    # Test it works with multiple elements and a funky shape.
    # pylint: disable=too-many-function-args
    [np.array([0.75,0.1875,1.5], np.float32).reshape((1,3,1)), popxl.float8_143,  0, False, np.array([0b00111100, 0b00101100, 0b01000100], popxl.utils.np_dtype_float8_143).reshape(1,3,1)],
    # pylint: enable=too-many-function-args
])
def test_host_pow2scale_then_cast(src, dtype, log2_scale, nan_on_overflow, exp_vals):
    """ Test host_cast_pow2scale conversion for float8. """
    out = host_pow2scale_then_cast(src, dtype, log2_scale=log2_scale, nan_on_overflow=nan_on_overflow)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, exp_vals)

@pytest.mark.parametrize("dtype_from", [popxl.float32, popxl.float64])
@pytest.mark.parametrize("dtype_to", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("shape", [(1, 5), (16,),  (3, 4), (1,), (2, 9, 2, 1)])
@pytest.mark.parametrize("log2_scale", [-1, 0, 1])
def test_host_cast_then_pow2scale(dtype_from, dtype_to, shape, log2_scale):
    """ Test cast_then_pow2scale by generating random data, converting it to
        float8, then converting it back again and checking the result is
        close-ish. Effectively we're testing cast_then_pow2scale is the
        inverse of host_pow2scale_then_cast.
    """
    # Generate some random data.
    x = np.random.random(size=np.prod(shape)).reshape(shape).astype(dtype_from.as_numpy())
    # Convert to float8 data (this is tested already in
    # test_host_pow2scale_then_cast).
    y = host_pow2scale_then_cast(x, dtype=dtype_to, log2_scale=log2_scale, nan_on_overflow=False)
    # Convert it back again (negate the scale).
    z = host_cast_then_pow2scale(y, dtype_from, log2_scale=-log2_scale)
    np.testing.assert_allclose(x, z, rtol=0.1, atol=0.06)

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
    else:
        v = dtype(1)
        t = [[v, v], [v, v]]

    result = to_numpy(t, copy=copy)
    if src == "pt":
        t_np = t.detach().numpy()
    elif src == "np":
        t_np = t
    else:
        t_np = np.asarray(t)
    should_alias = not copy and not (t_np.dtype in downcast_np_dtypes.keys())
    assert np.shares_memory(t_np, result) == should_alias


@pytest.mark.parametrize(("dtype", "result_dtype"), downcast_np_dtypes.items())
def test_to_numpy_downcast(dtype, result_dtype):
    t = np.ones((2, 2), dtype=dtype)
    assert to_numpy(t).dtype == result_dtype
