# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from enum import Enum

import popart._internal.ir as _ir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popxl import Ir


class TensorStorage(Enum):
    """Enum type to specify whether to store tensors on-chp in tile memory, or in Streaming Memory."""
    OnChip = "OnChip"  #: Store the tensor in tile memory.
    OffChip = "OffChip"  #: Store the tensor in Streaming Memory.

    @classmethod
    def _from_ir(cls, _ir_rts: _ir.TensorStorage):
        if _ir_rts == _ir.TensorStorage.OffChip:
            return super().__new__(cls, TensorStorage.OffChip)
        return super().__new__(cls, TensorStorage.OnChip)

    def _to__ir(self) -> _ir.TensorStorage:
        if self.OffChip:
            return _ir.TensorStorage.OffChip
        else:
            return _ir.TensorStorage.OnChip


class ReplicatedTensorSharding(Enum):
    """Enum type to specify whether to shard tensors over replicas."""
    On = "On"  #: Tensors will be sharded over replicas.
    Off = "Off"  #: Tensors will not be sharded over replicas.

    @classmethod
    def _from_ir(cls, _ir_rts: _ir.ReplicatedTensorSharding):
        if _ir_rts == _ir.ReplicatedTensorSharding.On:
            return super().__new__(cls, ReplicatedTensorSharding.On)
        return super().__new__(cls, ReplicatedTensorSharding.Off)

    def _to__ir(self) -> _ir.ReplicatedTensorSharding:
        if self.On:
            return _ir.ReplicatedTensorSharding.On
        else:
            return _ir.ReplicatedTensorSharding.Off


TileSet = _ir.TileSet
ExecutionContext = _ir.ExecutionContext


class TensorLocation():
    """Class that describes the memory characteristics of one or multiple tensors."""

    def __init__(
            self,
            storage: TensorStorage = TensorStorage.OnChip,
            replicated_tensor_sharding:
            ReplicatedTensorSharding = ReplicatedTensorSharding.Off) -> None:
        self._storage = storage
        self._replicated_tensor_sharding = replicated_tensor_sharding

    @classmethod
    def _from__ir(cls, _ir_tensor_location: _ir.TensorLocation):
        self = super().__new__(cls)
        self._storage = TensorStorage._from__ir(_ir_tensor_location.storage)
        self._replicated_tensor_sharding = ReplicatedTensorSharding._from__ir(
            _ir_tensor_location.replicatedTensorSharding)
        return self

    @property
    def storage(self) -> TensorStorage:
        return self._storage

    @property
    def replicated_tensor_sharding(self) -> ReplicatedTensorSharding:
        return self._replicated_tensor_sharding

    def _to__ir_location(self) -> _ir.TensorLocation:
        return _ir.TensorLocation(self.storage._to__ir,
                                  self.replicated_tensor_sharding._to__ir())
