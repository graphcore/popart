# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popxl.ir import Ir


class ReplicaGrouping:
    def __init__(self):
        """ Not intended to be called directly, use :py:func:`~popxl.Ir.replica_grouping` instead."""
        raise RuntimeError(
            "popxl.ReplicaGrouping cannot be constructed directly (use ir.replica_grouping)."
        )

    @classmethod
    def _from_params(cls, ir: 'Ir', stride: int,
                     group_size: int) -> 'ReplicaGrouping':
        self = super().__new__(cls)
        self._ir = ir
        self._stride = stride
        self._group_size = group_size
        return self

    @property
    def stride(self):
        """
        Get the stride.

        Returns:
            int: The offset between elements in a replica group.
        """
        return self._stride

    @property
    def group_size(self) -> int:
        """
        Get the group size.

        Returns:
            int: The number of replicas in each replica group.
        """
        return self._group_size

    def __repr__(self) -> str:
        """
        Returns a string representation.

        Returns:
            str: A string representation of this ReplicaGrouping instance.
        """
        return f"ReplicaGrouping(num_replicas={self._ir.replication_factor}, stride={self.stride}, group_size={self.group_size})"
