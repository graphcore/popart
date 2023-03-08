# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
Module containing tests and helper functions.
"""
from typing import Iterable, List
import random
import numpy as np
import popxl
from popxl import float8_143, float8_152
from popxl.utils import host_pow2scale_cast_to_fp8


def get_float8_data(
    float8_format: popxl.dtype, log2_scale: int, shape: Iterable[int]
) -> np.ndarray:
    """Generate some random float8 data."""
    d1 = get_representable_float_8_np_array(shape, float8_format, log2_scale)

    d1_float8 = host_pow2scale_cast_to_fp8(
        d1.astype(np.float32),
        float8_format,
        log2_scale,
        True,
    )

    return d1_float8


def generate_random_byte_array(allow_nans: bool = True) -> List[int]:
    """Generate a list of 0s and 1s, length 8"""

    def gen() -> List[int]:
        """Regenerate the list of random bits."""
        return [random.getrandbits(1) for _ in range(8)]

    array_ = gen()
    if not allow_nans:
        nan_values = [[1, 0, 0, 0, 0, 0, 0, 0]]
        while array_ in nan_values:
            # regen array
            array_ = gen()

    return array_


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
    given the provided float8 format and log2_scale.
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
    shape: Iterable[int],
    float8_format: popxl.dtype,
    log2_scale: int = 0,
    allow_nans=True,
) -> np.ndarray:
    """Get an array of given shapes with float16s that are all representable in
    the given float8 format and log2_scale"""
    array: List[float] = []
    for _ in range(np.prod(shape)):
        array.append(
            get_float8_decimal_from_byte_array(
                generate_random_byte_array(allow_nans), float8_format, log2_scale
            )
        )
    return np.array(array, np.float16).reshape(shape)
