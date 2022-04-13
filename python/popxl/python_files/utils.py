# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Union, Optional, TYPE_CHECKING

import numpy as np
import popart
from typing_extensions import Literal

try:
    import torch
    torch_imported = True
except ModuleNotFoundError:
    torch_imported = False

if TYPE_CHECKING:
    from popxl.tensor import Constant, Variable, dtypes, HostScalarTensor

HW_DEVICE_CONNECTION_TIMEOUT = int(1e4)

downcast_np_dtypes = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
}


def to_numpy(
        x: 'HostScalarTensor',
        dtype: Optional['dtypes.dtype'] = None,
        downcast: bool = True,
        copy: bool = True,
) -> np.ndarray:
    """
    Convert a `HostScalarTensor` to a numpy array and copies the data if enabled.

    Args:
        x (HostScalarTensor):
            The data used to initialise the tensor.
            This can be an np.ndarray, torch.tensor or a value NumPy can use to construct an np.ndarray.
        dtype (Optional[dtype]):
            The data type of the tensor to be created. If not specified NumPy will infer the data
            type and downcast to 32 bits if necessary.
        downcast (bool):
            If True and no dtype is provided, 64-bit float/ints will be downcast to 32-bit variants. Defaults to True.
        copy (bool):
            If true the objects data is guaranteed to be copied.
    """

    if dtype:
        np_dtype = dtype.as_numpy() if dtype is not None else None

        if np_dtype in downcast_np_dtypes and downcast and dtype is None:
            np_dtype = downcast_np_dtypes[np_dtype]
    else:
        np_dtype = None

    if torch_imported and isinstance(x, torch.Tensor):
        x = x.detach().numpy()
        if dtype:
            x = x.astype(np_dtype)
    else:
        x = np.array(x, dtype=np_dtype)

    if x.dtype in downcast_np_dtypes and downcast and dtype is None:
        x = x.astype(downcast_np_dtypes[x.dtype])
    elif copy:
        x = x.copy()

    return x


def _popxl_to_numpy(t: Union['Constant', 'Variable']) -> np.ndarray:
    """Return the data contained in the tensor. See the cpp class `TensorData` for details.

    Note this is a memory view of the data, so will not allocate extra memory for the data, but
    it is your responsibility to ensure the data in the tensor is live at the point
    of retrival.

    Args:
        t (Tensor): The tensor to retrive the data from.

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
        return np.asarray(t_.dataAsFloat16().view(t.dtype.as_numpy()),
                          t.dtype.as_numpy())
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
    else:
        raise ValueError(
            f"Data type {t.dtype} not supported for get_tensor_data retrival.")


def _to_device_info(device_type: Literal["ipu_hw", "ipu_model", "cpu"],
                    num_ipus: int = 1) -> popart.DeviceInfo:
    """Return the PopART `DeviceInfo` object relating to the given parameters.

    Args:
        device_type (Literal[str]): One of:
            "ipu_hw": Real IPU hardware. Uses DeviceConnectionType == OnDemand and
                DeviceSelectionCriterion == Random.
            "ipu_model": IPU model.
            "cpu": CPU model. Does not support replication.
        num_ipus (int, optional): Number of ipus to request. Defaults to 1.

    Raises:
        ValueError: If device_type == "cpu" and num_ipus > 1. This is not supported.
        ValueError: device_type is not one of "ipu_hw", "ipu_model", "cpu"

    Returns:
        popart.DeviceInfo: The device info for the given options.
    """

    if device_type == "cpu":
        if num_ipus > 1:
            raise ValueError(f"For a cpu device, multiple devices "
                             f"(provided: {num_ipus}) are not supported.")
        return popart.DeviceManager().createCpuDevice()
    elif device_type == "ipu_model":
        return popart.DeviceManager().createIpuModelDevice(
            {"numIPUs": num_ipus})
    elif device_type == "ipu_hw":
        # TODO: T56055 prompt the user that the session is waiting for a device.
        dm = popart.DeviceManager()
        dm.setOnDemandAttachTimeout(HW_DEVICE_CONNECTION_TIMEOUT)
        device = dm.acquireAvailableDevice(
            num_ipus,
            connectionType=popart.DeviceConnectionType.OnDemand,
            selectionCriterion=popart.DeviceSelectionCriterion.Random)
        return device
    raise ValueError(f"Incorrect device type provided: {device_type}, must be"
                     "one of: `hw`, `ipu_model`, `cpu`")