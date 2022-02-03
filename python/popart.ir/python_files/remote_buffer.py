# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popart.ir.context import gcg
from typing import Tuple
import popart._internal.ir as _ir
from popart.ir.dtypes import dtype
from popart.ir import Ir
from popart.ir.tensor import Tensor


class RemoteBuffer:
    """Handle to store to or load from remote buffers residing in the off-chip streaming memory.
    """

    def __init__(self, ir: Ir, tensor_shape: Tuple[int, ...],
                 tensor_dtype: dtype, entries: int) -> None:
        """Initialize the RemoteBuffer.

        This constructor will automatically assign a buffer id and store the buffer information in
        the underlying remoteBufferInfoMap.

        Args:
            ir (Ir): Ir where the remote buffer is to be set
            tensor_shape (Tuple[int, ...]]): The shape of the tensors stored in the buffer
            tensor_dtype (dtype) : The type stored in the buffer
            entries (int): The number of tensors with the same shape and type stored in the buffer

        Raises:
            ValueError: If entries < 1
        """
        if entries < 1:
            raise ValueError(
                f"Entries must be a non-zero, positive integer. Got {entries}")

        # Get the python bound ir
        self._current_ir = ir._pb_ir
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
            tensor_shape (Tuple[int, ...]]): The shape of the tensors stored in the buffer
            tensor_dtype (dtype) : The type stored in the buffer
            entries (int): The number of tensors with the same shape and type stored in the buffer
        """
        # Set the remote buffer info map
        tensor_info = _ir.TensorInfo(tensor_dtype._pb_dtype, tensor_shape)
        remote_buffer_info = _ir.RemoteBufferInfo(tensor_info, entries)
        self._current_ir.setRemoteBufferInfo(self._remote_buffer_id,
                                             remote_buffer_info)

    def validate_tensor_matches_buffer(self, t: Tensor) -> None:
        """Validate whether the tensor information matches that of the buffer.

        Args:
            t (Tensor): Tensor to check

        Raises:
            ValueError: If the tensor does not match the buffer
        """
        remote_buffer_info = self._current_ir.getRemoteBufferInfo(
            self.remote_buffer_id)
        existing_shape = tuple(remote_buffer_info.TensorInfo.shape())
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
        """Return the id to the buffer.

        Note that the id is read only.

        Returns:
            int: The id to the buffer
        """
        return self._remote_buffer_id

    @property
    def tensor_shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensors stored in the buffer.

        Once set, the shape cannot be reset.

        Returns:
            Tuple[int, ...]: The shape of the tensors stored in the buffer
        """
        return tuple(
            self._current_ir.getRemoteBufferInfo(
                self._remote_buffer_id).TensorInfo.shape())

    @property
    def tensor_dtype(self) -> dtype:
        """Return the type stored in the buffer.

        Once set, the type cannot be reset.

        Returns:
            dtype (dtype) : The type stored in the buffer
        """
        return dtype.as_dtype(
            self._current_ir.getRemoteBufferInfo(
                self._remote_buffer_id).TensorInfo.dataType())

    @property
    def entries(self) -> int:
        """Return the number of entries in the buffer.

        The setters checks that the value is not a negative value, and resets the buffer.

        Returns:
            int: The number of tensors with the same shape and type stored in the buffer

        Raises:
            ValueError: If the input is a non-positive integer
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


def remote_buffer(tensor_shape: Tuple[int, ...], tensor_dtype: dtype,
                  entries: int) -> RemoteBuffer:
    """Return a remote buffer based on the current ir from the context.

    Args:
        tensor_shape (Tuple[int, ...]]): The shape of the tensors stored in the buffer
        tensor_dtype (dtype) : The type stored in the buffer
        entries (int): The number of tensors with the same shape and type stored in the buffer

    Returns:
        RemoteBuffer: The remote buffer based on the current ir from the context
    """
    return RemoteBuffer(gcg().ir(), tensor_shape, tensor_dtype, entries)
