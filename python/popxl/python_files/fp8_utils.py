# Copyright (c) 2023 Graphcore Ltd. All rights reserved.


import numpy as np
from collections import namedtuple, defaultdict
from popxl import float8_143, float8_152, float16, float32, float64
import popart._internal.ir as _ir
from popxl.dtypes import np_dtype_float8_143, np_dtype_float8_152
import math

try:
    import torch

    torch_imported = True
except ModuleNotFoundError:
    torch_imported = False

max_scaling_bias = 31
min_scaling_bias = -31


# Define each floating point format that we use
FpDef = namedtuple(
    "FpDef",
    [
        "num_exp_bits",  # Number of exponent bits
        "num_mantissa_bits",  # Number of mantissa bits
        "fix_exponent_bias",  # The bias value to add to the exponent
        "include_denorm",  # Do we consider denormalised values in the format?
        "has_nans",  # True if a whole exponent "code" is used to represent Nan
    ],
)

ieee_fp32_no_denorms = FpDef(8, 23, 127, False, True)
ieee_fp16_denorms = FpDef(5, 10, 15, True, True)
float8_152_def = FpDef(5, 2, 16, True, False)
float8_143_def = FpDef(4, 3, 8, True, False)


# find the exponential range without dynamic scale `b`, defined by `(p-e)`
def find_exponential_range(fp_def: FpDef):
    min = 1 - fp_def.fix_exponent_bias
    max = (1 << fp_def.num_exp_bits) - 1 - fp_def.has_nans - fp_def.fix_exponent_bias
    if fp_def.include_denorm:
        min -= fp_def.num_mantissa_bits
    return min, max


# Find the histogram of each exponent
def log_histogram(
    src: np.ndarray,
    ftype: FpDef,
):
    # The bins edges are defined as [0, 2bs, 2bs+1, …, 2es] where bs and es are the lowest and highest exponents
    # representable for the floating point format defined for `src`.
    min_exp, max_exp = find_exponential_range(ftype)
    limits = [2 ** i for i in range(min_exp, max_exp)]
    # The histogram also includes underflow and overflow that are outside of the limits
    histogram = np.zeros(len(limits) + 1)
    for data in src:
        data = math.fabs(data)
        if data < limits[0]:
            histogram[0] += 1
        elif data > limits[-1]:
            histogram[-1] += 1
        else:
            _, e = np.frexp(data)
            histogram[e - min_exp] += 1

    return histogram, limits


class CandidateConfig:
    def __init__(self, format, src_ftype, dest_ftype, src_type):
        self.format = format
        self.exponent_size = dest_ftype.num_exp_bits
        self.mantissa_size = dest_ftype.num_mantissa_bits
        self.fix_exponent_bias = dest_ftype.fix_exponent_bias
        error_lsb = (
            ieee_fp32_no_denorms.num_mantissa_bits
            if src_type == np.float32
            else ieee_fp16_denorms.num_mantissa_bits
        )
        error_lsb -= self.mantissa_size
        self.quantisation_error = 2 ** (-2.0 * error_lsb)
        # source data exponent index
        min_hist_exponent, max_hist_exponent = find_exponential_range(src_ftype)
        # dest data exponent index
        min_bias_exp, max_bias_exp = find_exponential_range(dest_ftype)
        self.format_span = max_bias_exp - min_bias_exp + 1

        # the min index of the exp of source data that can be represented in dest type
        if min_bias_exp + min_scaling_bias - min_hist_exponent > 0:
            self.min_src_exp_index = min_bias_exp + min_scaling_bias - min_hist_exponent
        else:
            self.min_src_exp_index = 0

        # the max index of the exp of source data that can be represented in dest type
        self.max_src_exp_index = min(
            min_bias_exp + max_scaling_bias - min_hist_exponent,
            max_hist_exponent - min_hist_exponent,
        )
        self.num_exp_src = self.max_src_exp_index - self.min_src_exp_index + 1


def calculate_metric_mse(histogram, limits, src_bias_starting_indices, candidate):
    format_span, quantisation_error = (
        candidate.format_span,
        candidate.quantisation_error,
    )
    mse_index = defaultdict(list)
    for index in range(len(src_bias_starting_indices)):
        scaling_bias = index + min_scaling_bias
        min_index_src = src_bias_starting_indices[index]
        mse = 0
        # underflow
        for i in range(0, min_index_src):
            if i == 0:
                mse += (limits[0] / 2) ** 2
            mse += (limits[i] ** 2) * histogram[i + 1]
        # quantisation error
        for i in range(min_index_src, min(min_index_src + format_span, len(limits))):
            mse += limits[i] ** 2 * histogram[i + 1] * quantisation_error
        # overflow
        for i in range(min_index_src + format_span, len(limits)):
            mse += (limits[index + format_span - 1] - limits[i]) ** 2 * histogram[i + 1]
        print(f"mse {mse} scaling bias {scaling_bias}")
        mse_index[mse].append(scaling_bias)

    lowest_mse = sorted(mse_index)[0]
    candidate_indices = mse_index[lowest_mse]
    best_scaling_bias = sorted(candidate_indices)[len(candidate_indices) // 2]

    return lowest_mse, best_scaling_bias


def calculate_candidate_metric(hist_norm, limits, candidate, metric_type=None):
    # the starting indices of src bias that can be represented by the dest type, format span will be added during calculation
    src_bias_indices = np.arange(
        candidate.min_src_exp_index, candidate.min_src_exp_index + candidate.num_exp_src
    )
    # calculate the metric according to type
    if metric_type == "MSE":
        lowest_cost, best_index = calculate_metric_mse(
            hist_norm, limits, src_bias_indices, candidate
        )
    else:
        raise RuntimeError("Only MSE metric is supported.")
    return lowest_cost, best_index


def host_fp8_mse_scale(
    src: np.ndarray,
    dtype: "dtypes.dtype" = None,
) -> int:
    """
    Calculate the scale for a given float8 format automatically based on the MSE metrics.

    `s = host_fp8_mse_scale_pb(x, dtype)`

    computes `s` such that the cast

    .. code-block:: python

        x_scaled = x * (2 ** s)
        y = cast(x_scaled, dtype)

    loses the least information, i.e. "norm(y - x)" is minimized

    Terms definition:
    A float number is represented by `F*2**(p+b-e)`
    The `F` is represented by mantissa bits
    The `p` is represented by the exp bits
    The `e` is a scale that is fixed for each float type, called `fix_exponent_bias` in this code
    The `p-e` is in [-10, 7] for FP8-143, can be calculated by `find_exponential_range()`
    The `b` is represented by a 6 bit signed value. Its range is `[min_scaling_bias, max_scaling_bias]`, [-31, 31]
    This code is looking for the best scaling bias in `b` that minimizes the cost of FP8 format representing src data.

    Algorithm overview:
    1. Build a histogram with logarithmically spaced bin boundaries:
       2^-127, 2^-126 ...2^0, 2^1, ... 2^127 , using the numeric range of the
       input data format
    2. Normalise the histogram
    3. Define Candidates for each destination format, capturing
       the range the number format can represent without scale and
       the range of allowed scales for that format.
    4. For each format and the whole range of allowed scales measure
       the error if we try to represent the histogram's data with
       the specific format,scale.
    5. Pick the format and scale that has the lowest cost.
       Where cost is equal pick that with the largest number of mantissa bits
    6. Negate the scale to match PopART

    Args:
        src:
            A PopXL NumPy-based data array to convert to `dtype`. This must be a
            NumPy array with dtype float16 or float32.
        dtype:
            The PopXL dtype representing the target data type. This must be one
            of `popxl.float8_143` or `popxl.float8_152`.

    Raises:
        RuntimeError: If parameters are not supported.

    Returns:
        int: A scale calculated based on the input data `src`.
    """

    if src.dtype == np.float32 or src.dtype == np.float16:
        src_ftype = ieee_fp32_no_denorms
    else:
        raise RuntimeError(f"dtype {src.dtype} not currently supported.")

    dest_ftype = float8_143_def
    if dtype == float8_152:
        dest_ftype = float8_152_def
    src = np.ascontiguousarray(src).reshape(-1)
    hist, limits = log_histogram(src, src_ftype)
    hist_norm = hist / sum(hist)

    candidate = CandidateConfig(dtype, src_ftype, dest_ftype, src.dtype)
    metric, bias = calculate_candidate_metric(hist_norm, limits, candidate, "MSE")

    return metric, -bias


def _convert_popxl_float8_dtype_to_popart(np_dtype):
    """Convert a PopXL float8 dtype into the PopART equivalent."""
    float8_popxl_to_popart_dtype = {
        float8_143: _ir.DataType.FLOAT8_143,
        float8_152: _ir.DataType.FLOAT8_152,
    }
    return float8_popxl_to_popart_dtype[np_dtype]


def _convert_popxl_float8_dtype_to_numpy(np_dtype):
    """Convert a PopXL float8 dtype into the structured dtype we use for representing float8 data in NumPy."""
    float8_popxl_to_numpy_dtype = {
        float8_143: np_dtype_float8_143,
        float8_152: np_dtype_float8_152,
    }
    return float8_popxl_to_numpy_dtype[np_dtype]


def _convert_numpy_float8_dtype_to_popxl(np_dtype):
    """Convert a structured dtype we use for representing float8 data to a PopXL dtype."""
    if np_dtype == np_dtype_float8_143:
        return float8_143
    elif np_dtype == np_dtype_float8_152:
        return float8_152
    else:
        raise RuntimeError("Not a float8 dtype ({np_dtype})")


def host_pow2scale_cast_to_fp8(
    src: np.ndarray,
    dtype: "dtypes.dtype" = None,
    log2_scale: int = 0,
    nan_on_overflow: bool = True,
):
    """
    Run a fused operation `cast(src * pow2(log2_scale), dtype)` on the host.

    This is a host-based utility function mainly intended to convert user data
    into PopXL's NumPy-based representation for float8 data.

    Args:
        src:
            The NumPy array of user data to convert. Torch tensors are
            automatically converted. This must be a NumPy array with dtype being
            `np.float16`, `np.float32` or `np.float64` (or torch equivalent).
            Other values are not supported.
        dtype:
            The PopXL data type to convert to. This must be either  either
            `popxl.float8_143` or `popxl.float8_152`. Other values are not
            currently supported.
        log2_scale:
            The user's data is multiplied by `pow2(log2_scale)` before casting.
            This must be an int in the range [-32, 32). Other values are not
            currently supported.
        nan_on_overflow:
            If set, replace values that cannot be represented by the requested
            dtype with np.nan values.

    Raises:
        RuntimeError: If parameters are not supported.

    Returns:
        np.ndarray: A NumPy array with structured dtype
            `popxl.utils.np_dtype_float8_143`
            (`np.dtype([("float8_143", "u1")])`) or
            `popxl.utils.np_dtype_float8_152`
            (`np.dtype([("float8_152", "u1")])`) containing float8 data.
    """
    if torch_imported and isinstance(src, torch.Tensor):
        src = src.detach().numpy()

    if not isinstance(src, np.ndarray):
        src = np.asarray(src, order="C")

    if dtype != float8_143 and dtype != float8_152:
        raise RuntimeError(f"dtype {dtype} not currently supported.")

    popart_dtype = _convert_popxl_float8_dtype_to_popart(dtype)
    if src.dtype == np.float16:
        res = _ir.convertFromFloat16ToFloat8AsUInt8(
            popart_dtype, src, log2_scale, nan_on_overflow
        )
    elif src.dtype == np.float32:
        res = _ir.convertFromFloat32ToFloat8AsUInt8(
            popart_dtype, src, log2_scale, nan_on_overflow
        )
    elif src.dtype == np.float64:
        res = _ir.convertFromFloat64ToFloat8AsUInt8(
            popart_dtype, src, log2_scale, nan_on_overflow
        )
    else:
        raise RuntimeError(f"src.dtype {src.dtype} not currently supported.")
    res = res.reshape(src.shape)
    np_dtype = _convert_popxl_float8_dtype_to_numpy(dtype)
    res = res.astype(dtype=np_dtype)
    return res


def host_pow2scale_cast_from_fp8(
    src: np.ndarray, dtype: "dtypes.dtype" = None, log2_scale: int = 0
):
    """
    Run a fused operation `cast(X, dtype) * pow2(log2_scale)` on the host.

    This is a host-based utility function mainly intended to convert into
    PopXL's NumPy-based representation for float8 data back into user data.

    Args:
        src:
            A PopXL NumPy-based float8 data array to convert. This must be a
            NumPy array with with structured dtype
            `popxl.utils.np_dtype_float8_143`
            (`np.dtype([("float8_143", "u1")])`) or
            `popxl.utils.np_dtype_float8_152`
            (`np.dtype([("float8_152", "u1")])`). Other values are not
            currently supported.
        dtype:
            The PopXL dtype representing the target array type. This must be one
            of `popxl.float16`, `popxl.float32` or `popxl.float64`. Other values
            are not currently supported.
        log2_scale:
            The data is multiplied by `pow2(log2_scale)` after casting. This
            must be an int in the range [-32, 32). Other values are not
            currently supported.

    Raises:
        RuntimeError: If parameters are not supported.

    Returns:
        np.ndarray: A NumPy array with dtype `np.float16`, `np.float32` or `np.float64`.
    """
    if src.dtype != np_dtype_float8_143 and src.dtype != np_dtype_float8_152:
        raise RuntimeError(f"src.dtype {src.dtype} not currently supported.")

    popxl_src_dtype = _convert_numpy_float8_dtype_to_popxl(src.dtype)
    popart_src_dtype = _convert_popxl_float8_dtype_to_popart(popxl_src_dtype)

    # Drop the structured dtype.
    src = src.astype(np.uint8)

    # Call the appropriate conversion function.
    if dtype == float16:
        res = _ir.convertFromFloat8AsUInt8ToFloat16(
            popart_src_dtype, src, np.dtype("float16"), log2_scale
        )
    elif dtype == float32:
        res = _ir.convertFromFloat8AsUInt8ToFloat32(
            popart_src_dtype, src, np.dtype("float32"), log2_scale
        )
    elif dtype == float64:
        res = _ir.convertFromFloat8AsUInt8ToFloat64(
            popart_src_dtype, src, np.dtype("float64"), log2_scale
        )
    else:
        raise RuntimeError(f"dtype {dtype} not currently supported.")

    res = res.reshape(src.shape)

    return res


def host_fp8_mse_scale_pb(
    src: np.ndarray,
    dtype: "dtypes.dtype" = None,
) -> int:
    """
    Calculate the scale for a given float8 format automatically based on the MSE metrics.

    `s = host_fp8_mse_scale_pb(x, dtype)`

    computes `s` such that the cast

    .. code-block:: python

        x_scaled = x * (2 ** s)
        y = cast(x_scaled, dtype)

    loses the least information, i.e. "norm(y - x)" is minimized

    Args:
        src:
            A PopXL NumPy-based data array to convert to `dtype`. This must be a
            NumPy array with dtype float16 or float32.
        dtype:
            The PopXL dtype representing the target data type. This must be one
            of `popxl.float8_143` or `popxl.float8_152`.

    Raises:
        RuntimeError: If parameters are not supported.

    Returns:
        int: A scale calculated based on the input data `src`.
    """
    if dtype != float8_143 and dtype != float8_152:
        raise RuntimeError(f"dtype {dtype} not currently supported.")

    popart_dtype = _convert_popxl_float8_dtype_to_popart(dtype)

    if torch_imported and isinstance(src, torch.Tensor):
        src = src.detach().numpy()
    if not isinstance(src, np.ndarray):
        src = np.asarray(src, order="C")

    res = 0

    # Negate the scale to match PopART
    if src.dtype == np.float16:
        res = -_ir.calculateScaleFromFloat16ToFloat8(src, popart_dtype)
    elif src.dtype == np.float32:
        res = -_ir.calculateScaleFromFloat32ToFloat8(src, popart_dtype)
    else:
        raise RuntimeError(f"src.dtype {src.dtype} not currently supported.")

    return res
