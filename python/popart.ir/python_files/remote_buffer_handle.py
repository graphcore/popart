# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Tuple, Optional
from popart.ir.dtypes import dtype

__all__ = ["RemoteBufferHandle"]


class RemoteBufferHandle:
    """Handle to use when either storing or loading from remote buffers."""
    # _buffer serves as a bookkeeper of created instances as we only want to have one instance per
    # remote_buffer_id
    _buffers: Dict[int, "RemoteBufferHandle"] = {}

    def __new__(cls,
                remote_buffer_id: Optional[int] = None,
                tensor_shape: Optional[Tuple[int, ...]] = None,
                tensor_dtype: Optional[dtype] = None,
                repeats: int = 1) -> "RemoteBufferHandle":
        """Create a new instance only if it hasn't been created before.

        Args:
            remote_buffer_id (Optional[int]): The id of the remote buffer.
              If none is given, an id will be assigned automatically.
            tensor_shape (Optional[Tuple[int, ...]], optional): The shape of the tensors stored in
              the buffer.
            tensor_dtype (Optional[dtype]) : The type stored in the buffer.
            repeats (int, optional): The number of tensors with the same shape and type stored
              in the buffer. Defaults to 1.

        Returns:
            RemoteBufferHandle : The instance corresponding to remote_buffer_id
        """
        existing_remote_buffers = cls._buffers.keys()

        if remote_buffer_id is None:
            if existing_remote_buffers:
                remote_buffer_id = max(existing_remote_buffers) + 1
            else:
                remote_buffer_id = 1

        if remote_buffer_id not in existing_remote_buffers:
            cls._buffers[remote_buffer_id] = super().__new__(cls)

        return cls._buffers[remote_buffer_id]

    def __init__(self,
                 remote_buffer_id: int = None,
                 tensor_shape: Optional[Tuple[int, ...]] = None,
                 tensor_dtype: Optional[dtype] = None,
                 repeats: int = 1) -> None:
        """Initialize the RemoteBufferHandle if not previously initialized.

        Args:
            remote_buffer_id (int): The id of the remote buffer.
            tensor_shape (Optional[Tuple[int, ...]], optional): The shape of the tensors stored in
              the buffer.
            tensor_dtype (Optional[dtype]) : The type stored in the buffer.
            repeats (int, optional): The number of tensors with the same shape and type stored
              in the buffer. Defaults to 1.

        Raises:
            NotImplementedError: If remote_buffer_id == -1
        """
        # Set attributes only if they have not been set before
        if not hasattr(self, "tensor_shape"):
            # Obtain the correct buffer id
            if remote_buffer_id is None:
                # If the remote buffer id is none, then __new__ has just created it
                remote_buffer_id = max(RemoteBufferHandle._buffers.keys())
            if remote_buffer_id == -1:
                raise NotImplementedError(
                    "remote_buffer_id = -1 (automatic RemoteSetup) not supported"
                )

            # Set member data
            self._remote_buffer_id = remote_buffer_id
            self._tensor_shape = tensor_shape
            self._tensor_dtype = tensor_dtype
            self._repeats = repeats

        # Call the setters
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.repeats = repeats

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
            Tuple[int, ...]: The shape of the tensors stored in the buffer.
        """
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, _tensor_shape: Optional[Tuple[int, ...]]
                     ) -> None:  # type: ignore
        if self._tensor_shape is None:
            self._tensor_shape = _tensor_shape
        elif _tensor_shape != self._tensor_shape:
            raise ValueError(
                f"Cannot reset buffer shape. Buffer {self.remote_buffer_id} is already set with shape {self.tensor_shape}"
            )

    @property
    def tensor_dtype(self) -> dtype:
        """Return the type stored in the buffer.

        Once set, the type cannot be reset.

        Returns:
            dtype (dtype) : The type stored in the buffer.
        """
        return self._tensor_dtype

    @tensor_dtype.setter
    def tensor_dtype(self,
                     _tensor_dtype: Optional[dtype]) -> None:  # type: ignore
        if self._tensor_dtype is None:
            self._tensor_dtype = _tensor_dtype
        elif _tensor_dtype != self._tensor_dtype:
            raise ValueError(
                f"Cannot reset buffer dtype. Buffer {self.remote_buffer_id} is already set with dtype {self.tensor_dtype}"
            )

    @property
    def repeats(self) -> int:
        """Return the number of tensors with the same shape and type stored in the buffer.

        The setters checks that the value is not a negative value.

        Returns:
            int: The number of tensors with the same shape and type stored in the buffer
        """
        return self._repeats

    @repeats.setter
    def repeats(self, _repeats: int) -> None:
        if _repeats < 1:
            raise ValueError(
                f"Repeats must be a non-zero, positive integer. Got {_repeats}"
            )
        else:
            self._repeats = _repeats
