# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import random
import numpy as np
import pytest
from typing import Iterable, List
from popxl.utils import host_pow2scale_then_cast, host_cast_then_pow2scale
import popxl
import popxl.ops as ops
from popxl import float8_143, float8_152


def get_float8_data(
    float8_format: popxl.dtype, log2_scale: int, shape: Iterable[int]
) -> np.ndarray:
    """Generate some random float8 data."""
    d1 = get_representable_float_8_np_array(shape, float8_format, log2_scale)

    d1_float8 = host_pow2scale_then_cast(
        d1.astype(np.float32),
        float8_format,
        log2_scale,
        True,
    )

    return d1_float8


def generate_random_byte_array() -> List[int]:
    """Generate a list of 0s and 1s, length 8"""
    return [random.getrandbits(1) for _ in range(8)]


def to_int(lst) -> int:
    """Get the int that a list of 0s and 1s represent as binary."""
    num = 0
    for b in lst:
        num = 2 * num + b
    return num


def get_float8_decimal_from_byte_array(
    input_: List[int], float8_format: popxl.dtype, log2_scale: int = 0
) -> float:
    """From a given 8 long list of 0s and 1s, return the float this represents,
    given the provided float 8 format and log2_scale.
    """
    s = 1 if input_[0] == 0 else -1
    m = 1
    if float8_format == float8_143:
        format_bias = 8
        e = to_int(input_[1:5])
        for i in range(3):
            m += input_[5 + i] * pow(2, -i - 1)
        return s * m * pow(2, e + log2_scale - format_bias)

    elif float8_format == float8_152:
        format_bias = 16
        e = to_int(input_[1:6])
        for i in range(2):
            m += input_[6 + i] * pow(2, -i - 1)
        return s * m * pow(2, e + log2_scale - format_bias)

    raise TypeError(f"bad float8 format {float8_format}")


def get_representable_float_8_np_array(
    shape: Iterable[int], float8_format: popxl.dtype, log2_scale: int = 0
) -> np.ndarray:
    """Get an array of given shapes with float16s that are all representable in
    the given float8 format and log2_scale"""
    array: List[float] = []
    for _ in range(np.prod(shape)):
        array.append(
            get_float8_decimal_from_byte_array(
                generate_random_byte_array(), float8_format, log2_scale
            )
        )
    return np.array(array, np.float16).reshape(shape)


@pytest.mark.parametrize(
    "float8_format",
    [float8_143, float8_152],
)
@pytest.mark.parametrize(
    "log2_scale",
    [-4, -1, 0, 1, 4],
)
def test_pow2scale_then_cast(float8_format: popxl.dtype, log2_scale: int):
    """Test casting float16 to float8 on device vs host.

    Args:
        float8_format (popxl.dtype): Format 143 or 152
        log2_scale (int): log scaling to use
    """

    shape = [10, 10, 10]

    # Note this array may contain the equivalent nan value for the given format,
    # so we must set nan_on_overflow = True on host.
    d1 = get_representable_float_8_np_array(shape, float8_format, log2_scale)

    d2 = np.array(log2_scale, np.int8)
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        input0 = popxl.h2d_stream(
            d1.shape,
            popxl.float16,
            name="in_stream_0",
        )
        input1 = popxl.h2d_stream(
            d2.shape,
            popxl.int8,
            name="in_stream_0",
        )
        input_ = ops.host_load(input0, "input_")
        log2_scale_tensor = ops.host_load(input1, "log2_scale_tensor")

        cast = ops.pow2scale_then_cast(input_, log2_scale_tensor, float8_format)

        o_d2h = popxl.d2h_stream(cast.shape, cast.dtype, name="out_stream")
        ops.host_store(o_d2h, cast)
        with popxl.Session(ir, "ipu_model") as session:
            for i in range(5):
                print(f"Step {i}")

                outputs = session.run({input0: d1.astype(np.float16), input1: d2})

                # These should match exactly as the host conversion should be the same as
                # the device conversion.
                # 1. uint8 tensor returned from device, converted to popxl float8 numpy type
                array_1 = outputs[o_d2h]
                # 2. Original fp32 data converted on the host as popxl float8 numpy type
                array_2 = host_pow2scale_then_cast(
                    d1.astype(np.float32),
                    float8_format,
                    log2_scale,
                    True,
                )
                np.testing.assert_equal(array_1, array_2)

                # regenerate array
                d1 = get_representable_float_8_np_array(
                    shape, float8_format, log2_scale
                )


@pytest.mark.parametrize(
    "float8_format",
    [float8_143, float8_152],
)
@pytest.mark.parametrize(
    "log2_scale",
    [-4, -1, 0, 1, 4],
)
def test_cast_then_pow2scale(float8_format: popxl.dtype, log2_scale: int):
    """Test casting float8 to float16 on device vs host.

    Args:
        float8_format (popxl.dtype): Format 143 or 152
        log2_scale (int): log scaling to use
    """

    shape = [10, 10, 10]
    # Convert to float8 data on host. This is the baseline data we use for
    # both host and device conversion.

    # Note this array may contain the equivalent nan value for the given format,
    # so we must set nan_on_overflow = True on host.
    d1_float8 = get_float8_data(float8_format, log2_scale, shape)

    # When converting back again we will negate the scale.
    d2 = np.array(-log2_scale, np.int8)
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        input0 = popxl.h2d_stream(
            d1_float8.shape,
            float8_format,
            name="in_stream_0",
        )
        input1 = popxl.h2d_stream(
            d2.shape,
            popxl.int8,
            name="in_stream_0",
        )
        input_ = ops.host_load(input0, "input_")
        log2_scale_tensor = ops.host_load(input1, "log2_scale_tensor")

        cast = ops.cast_then_pow2scale(input_, log2_scale_tensor, popxl.float16)

        o_d2h = popxl.d2h_stream(cast.shape, cast.dtype, name="out_stream")
        ops.host_store(o_d2h, cast)
        with popxl.Session(ir, "ipu_model") as session:
            for i in range(5):
                print(f"Step {i}")
                outputs = session.run({input0: d1_float8, input1: d2})

                # These should match exactly as the host conversion should be the same as
                # the device conversion.
                # 1. Float16 data returned to the host from device
                array_1 = outputs[o_d2h].astype(np.float32)
                # 2. The float 8 array converted back to float32 on host.
                array_2 = host_cast_then_pow2scale(
                    d1_float8, popxl.float32, -log2_scale
                )
                np.testing.assert_equal(array_1, array_2)

                # regenerate array
                d1_float8 = get_float8_data(float8_format, log2_scale, shape)
