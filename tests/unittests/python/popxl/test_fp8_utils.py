# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import torch
import popxl

from popxl.fp8_utils import (
    host_pow2scale_cast_to_fp8,
    host_pow2scale_cast_from_fp8,
    host_fp8_mse_scale_pb,
    host_fp8_mse_scale,
)


@pytest.mark.parametrize("dtype_from", [popxl.float16, popxl.float32])
@pytest.mark.parametrize("dtype_to", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("shape", [(1, 5), (16,), (3, 4), (1,), (2, 9, 2, 1)])
def test_host_fp8_mse_scale_pb(dtype_from, dtype_to, shape):
    # Generate some random data.
    x = (
        np.random.random(size=np.prod(shape))
        .reshape(shape)
        .astype(dtype_from.as_numpy())
    )
    res = host_fp8_mse_scale_pb(x, dtype=dtype_to)
    if dtype_to == popxl.float8_143:
        assert -31 <= res <= 31
    if dtype_to == popxl.float8_152:
        assert -31 <= res <= 31


@pytest.mark.parametrize("dtype_from", [popxl.float16, popxl.float32])
def test_host_fp8_mse_scale(dtype_from):
    x = np.array([0.00098, 240]).astype(dtype_from.as_numpy())
    _, res = host_fp8_mse_scale(x, dtype=popxl.float8_143)
    assert res == 0

    x = np.array([2 ** (-17), 57344]).astype(dtype_from.as_numpy())
    _, res = host_fp8_mse_scale(x, dtype=popxl.float8_152)
    assert res == 0

    # If the data distribution is closer to overflow, the calculated scaling bias
    # should be negative: by multiplying by pow2(log2_scale), with log2_scale < 0,
    # the uniform data distribution between [2^6, 2^7] shifts towards the center
    # of the float8_143 dynamic range [2^-10, 2^7].
    x = np.random.uniform(2 ** 5, 2 ** 6, 1000).astype(dtype_from.as_numpy())
    _, res = host_fp8_mse_scale(x, dtype=popxl.float8_143)
    assert res < 0

    # Conversely, if the data distribution is closer to underflow, the calculated
    # scaling bias should be positive
    x = np.random.uniform(2 ** (-5), 2 ** (-6), 1000).astype(dtype_from.as_numpy())
    _, res = host_fp8_mse_scale(x, dtype=popxl.float8_143)
    assert res > 0


@pytest.mark.parametrize("dtype_from", [popxl.float16, popxl.float32])
@pytest.mark.parametrize("dtype_to", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("shape", [(1, 5), (16,), (3, 4), (1,), (2, 9, 2, 1)])
def test_host_calculate_pow2scale(dtype_from, dtype_to, shape):
    # Generate some random data.
    x = (
        np.random.random(size=np.prod(shape))
        .reshape(shape)
        .astype(dtype_from.as_numpy())
    )
    res = host_fp8_mse_scale_pb(x, dtype=dtype_to)
    if dtype_to == popxl.float8_143:
        assert -31 <= res <= 31
    if dtype_to == popxl.float8_152:
        assert -31 <= res <= 31


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
    # Test log2_scale for float8_143 + float16.
    [np.array([0.75], np.float16), popxl.float8_143,  0, False, np.array([0b00111100], popxl.dtypes.np_dtype_float8_143)],
    [np.array([0.75], np.float16), popxl.float8_143, -2, False, np.array([0b00101100], popxl.dtypes.np_dtype_float8_143)],
    [np.array([0.75], np.float16), popxl.float8_143,  1, False, np.array([0b01000100], popxl.dtypes.np_dtype_float8_143)],
    # Test log2_scale for float8_152 + float16.
    [np.array([0.25], np.float16), popxl.float8_152,  0, False, np.array([0b00111000], popxl.dtypes.np_dtype_float8_152)],
    [np.array([0.25], np.float16), popxl.float8_152, -3, False, np.array([0b00101100], popxl.dtypes.np_dtype_float8_152)],
    # Test log2_scale for float8_143 + float32.
    [np.array([0.75], np.float32), popxl.float8_143,  0, False, np.array([0b00111100], popxl.dtypes.np_dtype_float8_143)],
    [np.array([0.75], np.float32), popxl.float8_143, -2, False, np.array([0b00101100], popxl.dtypes.np_dtype_float8_143)],
    [np.array([0.75], np.float32), popxl.float8_143,  1, False, np.array([0b01000100], popxl.dtypes.np_dtype_float8_143)],
    # Test log2_scale for float8_152 + float32.
    [np.array([0.25], np.float32), popxl.float8_152,  0, False, np.array([0b00111000], popxl.dtypes.np_dtype_float8_152)],
    [np.array([0.25], np.float32), popxl.float8_152, -3, False, np.array([0b00101100], popxl.dtypes.np_dtype_float8_152)],
    # Test log2_scale for float8_143 + float64.
    [np.array([0.75], np.float64), popxl.float8_143,  0, False, np.array([0b00111100], popxl.dtypes.np_dtype_float8_143)],
    [np.array([0.75], np.float64), popxl.float8_143, -2, False, np.array([0b00101100], popxl.dtypes.np_dtype_float8_143)],
    # Test log2_scale for float8_152 + float64.
    [np.array([0.75], np.float64), popxl.float8_143,  1, False, np.array([0b01000100], popxl.dtypes.np_dtype_float8_143)],
    [np.array([0.25], np.float64), popxl.float8_152,  0, False, np.array([0b00111000], popxl.dtypes.np_dtype_float8_152)],
    [np.array([0.25], np.float64), popxl.float8_152, -3, False, np.array([0b00101100], popxl.dtypes.np_dtype_float8_152)],
    # Test nan_on_overflow for float8_143 + float16.
    [np.array([250.], np.float16), popxl.float8_143,  0, False, np.array([0b01111111], popxl.dtypes.np_dtype_float8_143)],
    [np.array([250.], np.float16), popxl.float8_143,  0, True,  np.array([0b10000000], popxl.dtypes.np_dtype_float8_143)],
    # Test nan_on_overflow for float8_152 + float16.
    [np.array([70000.], np.float16), popxl.float8_152,  0, False, np.array([0b01111111], popxl.dtypes.np_dtype_float8_152)],
    [np.array([70000.], np.float16), popxl.float8_152,  0, True,  np.array([0b10000000], popxl.dtypes.np_dtype_float8_152)],
    # Test nan_on_overflow for float8_143 + float32.
    [np.array([250.], np.float32), popxl.float8_143,  0, False, np.array([0b01111111], popxl.dtypes.np_dtype_float8_143)],
    [np.array([250.], np.float32), popxl.float8_143,  0, True,  np.array([0b10000000], popxl.dtypes.np_dtype_float8_143)],
    # Test nan_on_overflow for float8_152 + float32.
    [np.array([70000.], np.float32), popxl.float8_152,  0, False, np.array([0b01111111], popxl.dtypes.np_dtype_float8_152)],
    [np.array([70000.], np.float32), popxl.float8_152,  0, True,  np.array([0b10000000], popxl.dtypes.np_dtype_float8_152)],
    # Test nan_on_overflow for float8_143 + float64.
    [np.array([250.], np.float64), popxl.float8_143,  0, False, np.array([0b01111111], popxl.dtypes.np_dtype_float8_143)],
    [np.array([250.], np.float64), popxl.float8_143,  0, True,  np.array([0b10000000], popxl.dtypes.np_dtype_float8_143)],
    # Test nan_on_overflow for float8_152 + float64.
    [np.array([70000.], np.float64), popxl.float8_152,  0, False, np.array([0b01111111], popxl.dtypes.np_dtype_float8_152)],
    [np.array([70000.], np.float64), popxl.float8_152,  0, True,  np.array([0b10000000], popxl.dtypes.np_dtype_float8_152)],
    # Test it works with torch inputs.
    [torch.tensor([0.75], dtype=torch.float32), popxl.float8_143,  0, False, np.array([0b00111100], popxl.dtypes.np_dtype_float8_143)],
    # Test it works with multiple elements and a funky shape.
    # pylint: disable=too-many-function-args
    [np.array([0.75,0.1875,1.5], np.float32).reshape((1,3,1)), popxl.float8_143,  0, False, np.array([0b00111100, 0b00101100, 0b01000100], popxl.dtypes.np_dtype_float8_143).reshape(1,3,1)],
    # pylint: enable=too-many-function-args
])
def test_host_pow2scale_cast_to_fp8(src, dtype, log2_scale, nan_on_overflow, exp_vals):
    """ Test host_cast_pow2scale conversion for float8. """
    out = host_pow2scale_cast_to_fp8(src, dtype, log2_scale=log2_scale, nan_on_overflow=nan_on_overflow)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, exp_vals)

@pytest.mark.parametrize("dtype_from", [popxl.float16, popxl.float32, popxl.float64])
@pytest.mark.parametrize("dtype_to", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("shape", [(1, 5), (16,),  (3, 4), (1,), (2, 9, 2, 1)])
@pytest.mark.parametrize("log2_scale", [-1, 0, 1])
def test_host_pow2scale_cast_from_fp8(dtype_from, dtype_to, shape, log2_scale):
    """ Test pow2scale_cast_from_fp8 by generating random data, converting it to
        float8, then converting it back again and checking the result is
        close-ish. Effectively we're testing pow2scale_cast_from_fp8 is the
        inverse of host_pow2scale_cast_to_fp8.
    """
    # Generate some random data.
    x = np.random.random(size=np.prod(shape)).reshape(shape).astype(dtype_from.as_numpy())
    # Convert to float8 data (this is tested already in
    # test_host_pow2scale_cast_to_fp8).
    y = host_pow2scale_cast_to_fp8(x, dtype=dtype_to, log2_scale=log2_scale, nan_on_overflow=False)
    # Convert it back again (negate the scale).
    z = host_pow2scale_cast_from_fp8(y, dtype_from, log2_scale=-log2_scale)
    np.testing.assert_allclose(x, z, rtol=0.1, atol=0.06)
