# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import random
import struct
from typing import Callable, Tuple
import numpy as np
import popart
import pytest
from cmath import isnan, isinf
import popart._internal.ir as _ir


def float32_to_float8_143(
    fval: float, truncate: bool = False, scale: int = 0, exp_bias: int = 8
) -> int:
    """Convert a float32 value to a float8_143 (as int)
    By default, this conversion rounds-to-nearest-even and supports NaN and Inf
    Setting `truncate` to True enables a simpler conversion. In this mode the
    conversion simply dropping the 20 least significant bits of the significand.
    In this mode an error of up to 1 bit may be introduced.

    Args:
        fval (float): Value to convert.
        truncate (bool, optional): Whether to truncate overflows. Defaults to False.
        scale  (int, optional): The input is scaled by pow(2, scale_bias) before
            conversion to ensure input falls in the representable fp8 range. Defaults to 0.
        exp_bias (int, optional): Exponent bias to use. Defaults to 8.

    Returns:
        int: The converted fp8_143 value as an uint8.
    """
    # Scale value first
    fval = fval * pow(2, -scale)
    ival = int.from_bytes(struct.pack("<f", fval), "little")

    # NaN and Inf encode as -0
    if isnan(fval) or isinf(fval):
        return 0x80  # sign=1, exp=all-zeros, sig=0b000

    # round remaining bits using round-to-nearest-even
    if not truncate:
        round_ = ((ival >> 20) & 0x1) + 0x7FFFF
        ival = ival + round_

    # Overflow-then-inf, (1 + 7 / 8) * 2 ** ((2 ** 4 - 1) - exp_bias + scale)
    # Where:
    # (1 + 7 / 8) is the max manissa value.
    # (2 ** 4 - 1) is the maximal exponent
    max_val = (1 + 7 / 8) * pow(2, ((2 ** 4 - 1) - exp_bias + scale))
    if abs(fval) > max_val:
        return 0x80

    exp = ((ival & 0x7F800000) >> 23) - (2 ** 7 - 1) + exp_bias

    # underflow-then-zero
    if exp < 0:
        return 0x00

    return ((ival >> 24) & 0x80) | (exp << 3) | ((ival >> 20) & 0x7)


def float32_to_float8_152(
    fval: float, truncate: bool = False, scale: int = 0, exp_bias: int = 16
) -> int:
    """Convert a float32 value to a float8_152 (as int)
    By default, this conversion rounds-to-nearest-even and supports NaN and Inf
    Setting `truncate` to True enables a simper conversion. In this mode the
    conversion is simply dropping the 21 least significant bits of the significand.
    In this mode an error or up to 1 bit may be introduced.

    Args:
        fval (float): Value to convert.
        truncate (bool, optional): Whether to truncate overflows. Defaults to False.
        scale  (int, optional): The input is scaled by pow(2, scale_bias) before
            conversion to ensure input falls in the representable fp8 range.
            Defaults to 0.
        exp_bias (int, optional): Exponent bias to use. Defaults to 16.

    Returns:
        int: The converted fp8_152 value as an uint8.
    """
    # Scale value first
    fval = fval * pow(2, -scale)
    ival = int.from_bytes(struct.pack("<f", fval), "little")

    # NaN and Inf encode as -0
    if isnan(fval) or isinf(fval):
        return 0x80  # sign=1, exp=all-zeros, sig=0b00

    # round remaining bits using round-to-nearest-even
    if not truncate:
        round_ = ((ival >> 21) & 0x1) + 0xFFFFF
        ival = ival + round_

    # overflow-then-inf, 1.75 * (2 ** ((2 ** 5 -) 1 - exp_bias + scale))
    # Where:
    # (1 + 3 / 4) is the max manissa value.
    # (2 ** 5 - 1) is the maximal exponent.
    max_val = (1 + 3 / 4) * pow(2, ((2 ** 5 - 1) - exp_bias + scale))
    if abs(fval) > max_val:
        return 0x80

    exp = ((ival & 0x7F800000) >> 23) - (2 ** 7 - 1) + exp_bias

    # underflow-then-zero
    if exp < 0:
        return 0x00

    return ((ival >> 24) & 0x80) | (exp << 2) | ((ival >> 21) & 0x3)


@pytest.mark.parametrize(
    "format_", [popart.DataType.FLOAT8_143, popart.DataType.FLOAT8_152]
)
@pytest.mark.parametrize("dtype_from", [np.float32, np.float64])
@pytest.mark.parametrize("dtype_to", [np.float32, np.float64])
@pytest.mark.parametrize("scale_bias", [-1, 0, 1])
class TestFP8Conversion:
    def conversion_func_to(self, dtype_to: np.dtype) -> Callable:
        """Get the correct python fp8 conversion function (casting to fp8).

        Args:
            dtype_to (np.dtype): Data type to cast to.

        Raises:
            TypeError: If an unrecognised data type is passed.

        Returns:
            Callable: The correct conversion function.
        """
        if dtype_to == np.float32:
            return _ir.convertFromFloat8AsUInt8ToFloat32
        elif dtype_to == np.float64:
            return _ir.convertFromFloat8AsUInt8ToFloat64
        else:
            raise TypeError(f"Unsupported type {dtype_to}")

    @pytest.mark.parametrize("shape", [(16,), (1, 5), (3, 4), (1,), (2, 9, 2, 1)])
    def test_convertToFP8(
        self,
        format_: popart.DataType,
        shape: Tuple,
        dtype_from: np.dtype,
        dtype_to: np.dtype,
        scale_bias: int,
    ) -> None:
        """Test conversion of contiguous arrays.
        Args:
            format_ (popart.DataType): FP8_143 or FP8_152 format
            shape (Tuple): Shape to use.
            dtype_from (np.dtype): Numpy dtype to convert from.
            dtype_to (np.dtype):  Numpy dtype to convert back to.
            scale_bias (int): The input is scaled by pow(2, scale_bias) before
                conversion to ensure input falls in the representable fp8 range.

        Raises:
            TypeError: If an incorrect format is provided.
        """

        a1 = None
        if shape != ():
            a1 = np.random.random(size=np.prod(shape)).reshape(shape).astype(dtype_from)
        else:
            a1 = np.array(random.random())

        assert a1.flags["C_CONTIGUOUS"]

        b1 = _ir.convertToFloat8AsUInt8(format_, a1, scale_bias, False)

        # atol of 1 due to 1 bit potential error.
        if format_ == popart.DataType.FLOAT8_143:
            conv = np.array(
                [float32_to_float8_143(i, True, scale=scale_bias) for i in a1.flatten()]
            )
            np.testing.assert_allclose(b1, conv, rtol=0.1, atol=1)
        elif format_ == popart.DataType.FLOAT8_152:
            conv = np.array(
                [float32_to_float8_152(i, True, scale=scale_bias) for i in a1.flatten()]
            )
            np.testing.assert_allclose(b1, conv, rtol=0.1, atol=1)
        else:
            raise TypeError(f"Unsupported format {format_}")

        c1 = self.conversion_func_to(dtype_to)(format_, b1, scale_bias)

        np.testing.assert_allclose(a1, c1.reshape(shape), rtol=0.1, atol=0.06)

    @pytest.mark.parametrize("shape", [(3, 4), (2, 9, 2, 1)])
    def test_convertToFP8_non_contiguous(
        self,
        format_: popart.DataType,
        shape: Tuple,
        dtype_from: np.dtype,
        dtype_to: np.dtype,
        scale_bias: int,
    ) -> None:
        """Test conversion of non-contiguous arrays. Note there is some modification
        needed to outputs since the conversion functions flatten input.

        Args:
            format_ (popart.DataType): FP8_143 or FP8_152 format
            shape (Tuple): Shape to use.
            dtype_from (np.dtype): Numpy dtype to convert from.
            dtype_to (np.dtype):  Numpy dtype to convert back to.
            scale_bias (int): The input is scaled by pow(2, scale_bias) before
                conversion to ensure input falls in the representable fp8 range.

        Raises:
            TypeError: If an incorrect format is provided.
        """

        # Force non-contiguous with reshape and transpose.
        a1 = np.random.randn(*shape).T.astype(dtype_from)
        transposed_shape = a1.shape

        b1 = _ir.convertToFloat8AsUInt8(format_, a1, scale_bias, False)
        assert not a1.flags["C_CONTIGUOUS"]

        # atol of 1 due to 1 bit potential error.
        if format_ == popart.DataType.FLOAT8_143:
            conv = np.array(
                [float32_to_float8_143(i, True, scale=scale_bias) for i in a1.flatten()]
            )
            np.testing.assert_allclose(b1, conv, rtol=0.1, atol=1)
        elif format_ == popart.DataType.FLOAT8_152:
            conv = np.array(
                [float32_to_float8_152(i, True, scale=scale_bias) for i in a1.flatten()]
            )
            np.testing.assert_allclose(b1, conv, rtol=0.1, atol=1)
        else:
            raise TypeError(f"Unsupported format {format_}")

        c1 = self.conversion_func_to(dtype_to)(format_, b1, scale_bias)

        # Note: outputs from conversion functions are flattened, so we must reshape
        # again here to compare.
        np.testing.assert_allclose(
            a1, c1.reshape(transposed_shape), rtol=0.1, atol=0.06
        )
