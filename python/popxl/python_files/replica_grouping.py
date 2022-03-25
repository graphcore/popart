# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Optional

import popart._internal.ir as _ir
from popart import VariableSettings

if TYPE_CHECKING:
    from popxl.ir import Ir


class ReplicaGrouping:
    def __init__(self):
        """ Not intended to be called directly, use :py:func:`~popxl.Ir.replica_grouping` instead."""
        raise RuntimeError(
            "popxl.ReplicaGrouping cannot be constructed directly (use ir.replica_grouping)."
        )

    @classmethod
    def _from_params(cls,
                     ir: 'Ir',
                     stride: Optional[int] = 1,
                     group_size: Optional[int] = None) -> 'ReplicaGrouping':
        self = super().__new__(cls)
        self._ir = ir
        self._stride = stride
        if group_size is None:
            self._group_size = ir.replication_factor // stride
        else:
            self._group_size = group_size

        return self

    @property
    def stride(self) -> int:
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

    @property
    def num_groups(self) -> int:
        """
        Get the number of groups.

        Returns:
            int: The number of replica groups.
        """
        if self.group_size > 1:
            return self._ir.replication_factor // self._group_size
        return self._ir.replication_factor

    def __repr__(self) -> str:
        """
        Return a string representation.

        Returns:
            str: A string representation of this ReplicaGrouping instance.
        """
        return f"ReplicaGrouping(num_replicas={self._ir.replication_factor}, stride={self.stride}, group_size={self.group_size})"

    def _to_variable_settings(self) -> VariableSettings:
        """
        Create a popart.VariableSettings object from this ReplicaGrouping's stride and group_size.

        Raises:
            ValueError: If incompatible stride and group_size arguments are passed.

        Returns:
            VariableSettings: The popart equivalent of ReplicaGroupings.
        """
        replicas: int = self._ir.replication_factor
        comm_group: _ir.CommGroup

        variable_settings: VariableSettings = None
        # If replica_grouping.group_size==1 => use CommGroupType::None
        if self.group_size == 1:
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.Ungrouped,
                                       replicaGroupSize=0)

            variable_settings = VariableSettings(comm_group)
        # If replica_grouping.group_size==N => use CommGroupType::All
        elif self.group_size == replicas:
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.All,
                                       replicaGroupSize=0)
            variable_settings = VariableSettings(comm_group)
        # If replica_grouping.stride==1 (and replica_grouping.group_size divides N) =>
        # use CommGroupType::Consecutive with size=replica_grouping.group_size
        elif self.stride == 1 and (replicas % self.group_size == 0):
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.Consecutive,
                                       replicaGroupSize=self.group_size)
            variable_settings = VariableSettings(comm_group)
        # If replica_grouping.stride==N/replica_grouping.group_size =>
        # use CommGroupType::Orthogonal with size=replica_grouping.group_size
        elif (replicas % self.group_size == 0) and self.stride == (
                replicas // self.group_size):
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.Orthogonal,
                                       replicaGroupSize=self.group_size)
            variable_settings = VariableSettings(comm_group)
        else:
            raise ValueError(
                f"Replica grouping with num_replicas={replicas}, "
                f"stride={self.stride} and group_size={self.group_size} is not currently supported"
            )
        variable_settings.verify()
        return variable_settings
