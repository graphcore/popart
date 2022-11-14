# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import sys
import time
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import popart
import popart._internal.ir as _ir
from popxl import float8_143, float8_152, float16, float32, float64
from typing_extensions import Literal

from popxl.dtypes import np_dtype_float8_143, np_dtype_float8_152

try:
    import torch

    torch_imported = True
except ModuleNotFoundError:
    torch_imported = False

if TYPE_CHECKING:
    from popxl.tensor import Constant, HostScalarTensor, Variable, dtypes

HW_DEVICE_CONNECTION_TIMEOUT = int(1e4)

downcast_np_dtypes = {
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
    np.dtype("float64"): np.dtype("float32"),
}


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


def to_numpy(
    x: "HostScalarTensor",
    dtype: Optional["dtypes.dtype"] = None,
    downcast: bool = True,
    copy: bool = True,
    log2_scale: int = None,
    nan_on_overflow: bool = None,
) -> np.ndarray:
    """
    Convert a `HostScalarTensor` to a numpy array and copies the data if enabled.

    Args:
        x:
            The data used to initialise the tensor. This can be an np.ndarray,
            torch.tensor or a value NumPy can use to construct an np.ndarray. If
            dtype is of float8 type this must be a np.float16, np.float32 or
            float64 type, torch equivalent, or native type equivalent. Other
            values are not supported.
        dtype:
            The data type of the tensor to be created. If not specified NumPy
            will infer the data type and downcast to 32 bits if necessary. For
            float8 dtypes automatic inference of dtype is not currently
            possible, please explicitly specify the dtype.
        downcast:
            If True and no dtype is provided, 64-bit float/ints will be downcast
            to 32-bit variants. Defaults to True.
        copy:
            If true the objects data is guaranteed to be copied.
        log2_scale:
            If dtype is either popxl.float8_143 or popxl.float8_152 then
            multiply the incoming data by pow2(log2_scale) before casting.
        nan_on_overflow:
            If dtype is either popxl.float8_143 or popxl.float8_152 and this
            flag is set then replace values that cannot be represented by the
            requested dtype with np.nan values.

    Raises:
        RuntimeError: If parameters are not supported.

        TypeError: If dtype is of float8 type and x is not of type np.float16,
        np.float32 or float64, torch equivalent, or native type equivalent.

    Returns:
        np.ndarray: A NumPy array.
    """
    if torch_imported and isinstance(x, torch.Tensor):
        x = x.detach().numpy()

    # Work out the target dtype.
    if dtype == float8_143 or dtype == float8_152:

        # If unset, don't scale.
        if log2_scale is None:
            log2_scale = 0

        # If unset, use nans.
        if nan_on_overflow is None:
            nan_on_overflow = True

        # Convert scalars to numpy array.
        x = np.asarray(x, order="C")

        # There is no native float8 representation in Numpy currently so we use
        # structured dtypes instead. If not already converted, convert the
        # user's data to float8 automatically.
        if x.dtype != _convert_popxl_float8_dtype_to_numpy(dtype):
            # But if inferred type is not float64 or float32, throw an error.
            if x.dtype not in [np.float16, np.float32, np.float64]:
                raise TypeError(
                    f"Type {x.dtype} is not supported for float8 tensors."
                    "Please use float16, float32 or float64 types."
                )

            x = host_pow2scale_then_cast(x, dtype, log2_scale, nan_on_overflow)

        if copy:
            x = x.copy()

    else:

        # Throw if log2_scale is used but we ignore it.
        if log2_scale is not None:
            raise RuntimeError(
                "Non-default value for 'log2scale' is not supported for dtypes "
                " that are not popxl.float8_143 or popxl.float8_152"
            )

        # Throw if log2_scale is used but we ignore it.
        if nan_on_overflow is not None:
            raise RuntimeError(
                "Non-default value for 'nan_on_overflow' is not supported for "
                " dtypes that are not popxl.float8_143 or popxl.float8_152"
            )

        # If unset, use nans.
        if nan_on_overflow is not None:
            nan_on_overflow = True

        if dtype:
            np_dtype = dtype.as_numpy()
        elif isinstance(x, np.ndarray):
            np_dtype = x.dtype
        else:
            np_dtype = np.obj2sctype(x)

        # Attempt to downcast before constructing the array
        if not dtype and np_dtype in downcast_np_dtypes and downcast:
            np_dtype = downcast_np_dtypes[np_dtype]

        x = np.asarray(x, dtype=np_dtype, order="C")

        # Sometimes it is not possible to infer the result type before constructing the array
        # so check again here to ensure 64-bit values are not returned when downcast=True
        if not dtype and x.dtype in downcast_np_dtypes and downcast:
            x = np.array(x, dtype=downcast_np_dtypes[x.dtype], order="C")
        elif copy:
            x = x.copy()

    return x


def host_pow2scale_then_cast(
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


def host_cast_then_pow2scale(
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


def _popxl_to_numpy(t: Union["Constant", "Variable"]) -> np.ndarray:
    """Return the data contained in the tensor. See the cpp class `TensorData` for details.

    Note this is a memory view of the data, so will not allocate extra memory for the data, but
    it is your responsibility to ensure the data in the tensor is live at the point
    of retrieval.

    Args:
        t (Tensor): The tensor to retrieve the data from.

    Raises:
        ValueError: If the tensor is not of type [Variable|Constant]

    Returns:
        np.ndarray: A NumPy array containing the data from the tensor,
            with the same data type as the tensor.
    """
    from popxl.tensor import dtypes

    t_ = t._pb_tensor
    if t.dtype == dtypes.float64:
        return np.asarray(t_.dataAsFloat64(), t.dtype.as_numpy())
    elif t.dtype == dtypes.float32:
        return np.asarray(t_.dataAsFloat32(), t.dtype.as_numpy())
    elif t.dtype == dtypes.float16:
        # TODO T50782: Handle fp16 conversion in cpp and avoid the .view() call.
        # See python/popart._internal.ir/bindings/tensor.cpp
        return np.asarray(
            t_.dataAsFloat16().view(t.dtype.as_numpy()), t.dtype.as_numpy()
        )
    elif t.dtype == dtypes.int64:
        return np.asarray(t_.dataAsInt64(), t.dtype.as_numpy())
    elif t.dtype == dtypes.int32:
        return np.asarray(t_.dataAsInt32(), t.dtype.as_numpy())
    elif t.dtype == dtypes.int16:
        return np.asarray(t_.dataAsInt16(), t.dtype.as_numpy())
    elif t.dtype == dtypes.uint64:
        return np.asarray(t_.dataAsUInt64(), t.dtype.as_numpy())
    elif t.dtype == dtypes.uint32:
        return np.asarray(t_.dataAsUInt32(), t.dtype.as_numpy())
    elif t.dtype == dtypes.uint16:
        return np.asarray(t_.dataAsUInt16(), t.dtype.as_numpy())
    elif t.dtype == dtypes.double:
        return np.asarray(t_.dataAsDouble(), t.dtype.as_numpy())
    elif t.dtype == dtypes.bool:
        return np.asarray(t_.dataAsBool(), t.dtype.as_numpy())
    elif t.dtype == dtypes.float8_143:
        return np.asarray(t_.dataAsUInt8(), np_dtype_float8_143)
    elif t.dtype == dtypes.float8_152:
        return np.asarray(t_.dataAsUInt8(), np_dtype_float8_152)
    else:
        raise ValueError(
            f"Data type {t.dtype} not supported for get_tensor_data retrieval."
        )


def _offline_device_from_str(
    device_type: str, num_ipus: Optional[int] = None
) -> popart.DeviceInfo:
    """Return a PopART `DeviceInfo` object matching the name of a system.
        See poplar::Target::createIPUTarget for more details.

    Args:
        device_type (str): Name of an IPU system.
        num_ipus (int, optional): Total IPUs to use. Defaults to all IPUs in the named system.

    Returns:
        popart.DeviceInfo: The device info for the given options
    """
    return popart.DeviceManager().createOfflineIpuFromSystemString(
        device_type, num_ipus if num_ipus else 0
    )


def _to_device_info(
    device_type: Literal["ipu_hw", "ipu_model", "cpu"],
    num_ipus: int = 1,
    use_popdist: bool = False,
) -> popart.DeviceInfo:
    """Return the PopART `DeviceInfo` object relating to the given parameters.

    Args:
        device_type (Literal[str]): One of:
            "ipu_hw": OfflineIpuDevice matching the current system.
            "ipu_model": IPU model.
            "cpu": CPU model. Does not support replication.
        num_ipus (int, optional): Number of ipus to request. Defaults to 1.

    Raises:
        ValueError: If device_type == "cpu" and num_ipus > 1. This is not supported.
        ValueError: device_type is not one of "ipu_hw", "ipu_model", "cpu"
        RuntimeError: If there are not enough available IPUs

    Returns:
        popart.DeviceInfo: The device info for the given options.
    """
    if device_type == "cpu":
        if num_ipus > 1:
            raise ValueError(
                f"For a cpu device, multiple devices "
                f"(provided: {num_ipus}) are not supported."
            )
        return popart.DeviceManager().createCpuDevice()
    elif device_type == "ipu_model":
        return popart.DeviceManager().createIpuModelDevice({"numIPUs": num_ipus})
    elif device_type == "ipu_hw":
        if use_popdist:
            import popdist.popart

            return popdist.popart.getDevice(
                connectionType=popart.DeviceConnectionType.OnDemand
            )

        dm = popart.DeviceManager()
        devices = dm.enumerateDevices(numIpus=num_ipus)
        if len(devices) < 1:
            raise RuntimeError(
                f"Failed to acquire device with {num_ipus} IPUs. Ensure that there are "
                "sufficient IPUs available. If you have enabled the Poplar SDK you can "
                "check device availability with the `gc-monitor` command-line utility."
            )
        return dm.createOfflineIpuFromDeviceInfo(devices[0])
    try:
        # Allow passing system string for testing purposes.
        return _offline_device_from_str(device_type, num_ipus)
    except popart.exception:
        pass
    raise ValueError(
        f"Incorrect device type provided: {device_type}, must be "
        "one of: `ipu_hw`, `ipu_model`, `cpu`"
    )


def _print_acquired():
    # Check if stdout supports `utf-8` to avoid errors when printing
    if sys.stdout.encoding.lower() in {"utf-8", "utf_8"}:
        rocket = "\U0001F680"
    else:
        rocket = "device"
    print(f". Acquired {rocket}", end="\n")


def _print_waiting(elapsed, time_until_summary, update_frequency, devices):
    message = f"\r{devices} matching device{'s' if devices > 1 else ''}. Waiting for available device."
    dots = min(time_until_summary // update_frequency, elapsed // update_frequency)
    message += "." * int(dots)
    if elapsed >= time_until_summary:
        # After `time_until_summary` seconds just print the elapsed time instead of more dots
        message += f" waited {int(elapsed)}s"
    print(message, end="")


def _acquire_hw_device_with_timeout(
    num_ipus: int = 1, timeout: int = HW_DEVICE_CONNECTION_TIMEOUT
) -> popart.DeviceInfo:
    """Acquire a real IPU device with `num_ipus`.

    If there are matching devices available but they are busy, this method will wait for an available device.

    Args:
        num_ipus (int, optional): Number of IPUs required. Defaults to 1.
        timeout (int, optional): How many seconds to wait for matching devices to be available. Defaults to HW_DEVICE_CONNECTION_TIMEOUT.

    Raises:
        RuntimeError: If there are no matching devices
        RuntimeError: If timeout is reached before a matching device is available

    Returns:
        popart.DeviceInfo: Acquired device
    """
    devices = popart.DeviceManager().enumerateDevices(numIpus=num_ipus)
    if len(devices) < 1:
        raise RuntimeError(
            f"Failed to acquire device with {num_ipus} IPUs. Ensure that there are "
            "sufficient IPUs available. If you have enabled the Poplar SDK you can "
            "check device availability with the `gc-monitor` command-line utility."
        )
    start = time.time()
    elapsed = 0
    next_message = 0
    update_frequency = 1
    wait_time = 0.1
    time_until_summary_message = 15

    while elapsed <= timeout:
        for device in devices:
            device.attach()
            if device.isAttached:
                if elapsed > 0:
                    # Print only if we've waited for a device
                    _print_acquired()
                return device

        time.sleep(wait_time)

        elapsed = time.time() - start
        if elapsed > next_message:
            _print_waiting(
                elapsed, time_until_summary_message, update_frequency, len(devices)
            )
            next_message += update_frequency

    print(end="\n")
    raise RuntimeError(
        "Reached timeout waiting for an available device. Check `gc-monitor` for current device usage."
    )


def table_to_string(
    rows: Sequence[Sequence], delimiter: str = " | ", header: bool = True
):
    """Create a string that resembles a table from inputs `rows`.

    Each item in rows represents a row which will be delimited with `delimiter`.
    Each row should exactly have the same length.

    Example:

    .. code-block:: python

        rows = [
            ["num", "foo", "name"],
            [3, "aaab", "args"],
            [4, "barrrr", "kwargs"],
            [3, "me", "inspect"],
            [-1, "who", "popxl"],
        ]
        print(table_to_string(rows))

    Output:

    .. code-block:: none

        num | foo    | name
        -----------------------
        3   | aaab   | args
        4   | barrrr | kwargs
        3   | me     | inspect
        -1  | who    | popxl


    Args:
        rows (Sequence[Sequence]): A row by column nested sequence. Each item needs to be a string or stringable
        delimiter (str): String used to delimit columns
        header (bool): If true, the first row is underlined

    Returns:
        str: A string representation of the table
    """
    col_widths = [max(map(len, map(str, col))) for col in zip(*rows)]
    output = ""
    for i, row in enumerate(rows):
        row_str = (
            delimiter.join(f"{col: <{width}}" for col, width in zip(row, col_widths))
            + "\n"
        )
        output += row_str
        if header and i == 0 and len(rows) > 1:
            output += "-" * len(row_str) + "\n"

    return output
