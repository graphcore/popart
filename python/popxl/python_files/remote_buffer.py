# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popxl.context import gcg
from typing import Tuple
import popart._internal.ir as _ir
from popxl.dtypes import dtype
from popxl.tensor import Tensor


class RemoteBuffer:
    def __init__(self, tensor_shape: Tuple[int, ...], tensor_dtype: dtype,
                 entries: int) -> None:
        """Store to or load from remote buffers residing in Streaming Memory.

        This constructor will automatically assign a buffer ID and store the buffer information in
        the underlying `remoteBufferInfoMap`.

        Args:
            tensor_shape (Tuple[int, ...]]): The shape of the tensors to be stored in the buffer.
            tensor_dtype (dtype) : The type of the tensors to be stored in the buffer.
            entries (int): Sets the size of the buffer to this number of tensors with the specified shape and type.

        Raises:
            ValueError: If `entries` is a not a positive integer.
        """
        if entries < 1:
            raise ValueError(
                f"Entries must be a non-zero, positive integer. Got {entries}")

        # Get the python bound ir
        self._current_ir = gcg().ir._pb_ir
        self._meta_shape: Tuple[int, ...] = tuple()

        # Obtain the buffer id
        remote_buffer_info_map = self._current_ir.getAllRemoteBufferInfos()
        if len(remote_buffer_info_map.keys()) == 0:
            remote_buffer_id = 1
        else:
            remote_buffer_id = max(remote_buffer_info_map.keys()) + 1
            # Guard against negative buffer ids
            if remote_buffer_id < 0:
                remote_buffer_id = 1

        self._remote_buffer_id = remote_buffer_id

        # Set the remote buffer info
        self.set_remote_buffer_info(tensor_dtype=tensor_dtype,
                                    tensor_shape=tensor_shape,
                                    entries=entries)

    def set_remote_buffer_info(self, tensor_dtype: dtype,
                               tensor_shape: Tuple[int, ...],
                               entries: int) -> None:
        """Store the buffer information in the underlying remoteBufferInfoMap.

        Args:
            tensor_shape (Tuple[int, ...]]): The shape of the tensors to be stored in the buffer.
            tensor_dtype (dtype) : The type of the tensors to be stored in the buffer.
            entries (int): Sets the size of the buffer to this number of tensors with the specified shape and type.
        """
        # Set the remote buffer info map
        tensor_info = _ir.TensorInfo(tensor_dtype._pb_dtype, tensor_shape)
        remote_buffer_info = _ir.RemoteBufferInfo(tensor_info, entries)
        self._current_ir.setRemoteBufferInfo(self._remote_buffer_id,
                                             remote_buffer_info)

    def validate_tensor_matches_buffer(self, t: Tensor,
                                       num_shards: int = 1) -> None:
        """Validate whether the tensor information matches that of the buffer.

        Args:
            t (Tensor): Tensor to check.
            num_shards (int): The number of shards used.

        Raises:
            ValueError: If the tensor does not match the buffer.
        """
        remote_buffer_info = self._current_ir.getRemoteBufferInfo(
            self.remote_buffer_id)

        if num_shards == 1:
            existing_shape = tuple(remote_buffer_info.TensorInfo.shape())
        else:
            existing_shape = self.meta_shape
        existing_dtype = remote_buffer_info.TensorInfo.dataType()

        tensor_shape = t.shape
        tensor_dtype = t.dtype._pb_dtype
        if (tensor_shape != existing_shape) or (tensor_dtype !=
                                                existing_dtype):
            raise ValueError(f"Tensor does not match buffer.\n"
                             f"Existing remote buffer has "
                             f"shape={existing_shape} and "
                             f"dtype={existing_dtype}.\n"
                             f"The tensor has "
                             f"shape={tensor_shape}, "
                             f"dtype={tensor_dtype}.")

    @property
    def remote_buffer_id(self) -> int:
        """Return the ID of the buffer.

        Note that the ID is read only.

        Returns:
            int: The ID of the buffer.
        """
        return self._remote_buffer_id

    @property
    def tensor_shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensors stored in the buffer.

        The shape cannot be changed after the buffer has been created.

        Returns:
            Tuple[int, ...]: The shape of the tensors stored in the buffer.
        """
        return tuple(
            self._current_ir.getRemoteBufferInfo(
                self._remote_buffer_id).TensorInfo.shape())

    @property
    def tensor_dtype(self) -> dtype:
        """Return the type of the tensors stored in the buffer.

        The type cannot be changed after the buffer has been created.

        Returns:
            dtype (dtype): The type of the tensors stored in the buffer.
        """
        return dtype.as_dtype(
            self._current_ir.getRemoteBufferInfo(
                self._remote_buffer_id).TensorInfo.dataType())

    @property
    def entries(self) -> int:
        """Return the number of entries that can be stored in the buffer.

        Setting the value of this property will update the size of the buffer.

        Returns:
            int: The number of tensors with the specified shape and type that can be stored in the buffer.

        Raises:
            ValueError: If set to a value that is a not a positive integer.
        """
        return self._current_ir.getRemoteBufferInfo(
            self._remote_buffer_id).repeats

    @entries.setter
    def entries(self, _entries: int) -> None:
        if _entries < 1:
            raise ValueError(
                f"Entries must be a non-zero, positive integer. Got {_entries}"
            )
        else:
            self.set_remote_buffer_info(tensor_dtype=self.tensor_dtype,
                                        tensor_shape=self.tensor_shape,
                                        entries=_entries)

    @property
    def meta_shape(self) -> Tuple[int, ...]:
        return self._meta_shape

    @meta_shape.setter
    def meta_shape(self, shape: Tuple[int, ...]) -> None:
        self._meta_shape = shape

    def __hash__(self):
        return hash((self._current_ir, self.remote_buffer_id))


def remote_buffer(tensor_shape: Tuple[int, ...],
                  tensor_dtype: dtype,
                  entries: int = 1) -> RemoteBuffer:
    """Return a remote buffer based on the current IR from the context.

    Args:
        tensor_shape (Tuple[int, ...]]): The shape of the tensors to be stored in the buffer.
        tensor_dtype (dtype): The type of the tensors to be stored in the buffer.
        entries (int): Sets the size of the buffer to this number of tensors with the specified shape and type. Defaults to 1.

    Returns:
        RemoteBuffer: The remote buffer based on the current IR from the context.
    """
    return RemoteBuffer(tensor_shape, tensor_dtype, entries)
