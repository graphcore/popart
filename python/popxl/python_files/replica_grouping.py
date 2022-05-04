# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Optional
from typing_extensions import Literal

import popart._internal.ir as _ir
from popart import VariableSettings, VariableRetrievalMode

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

    @classmethod
    def _from_variable_settings(
            cls, ir: 'Ir',
            variable_settings: VariableSettings) -> "ReplicaGrouping":
        """Generate a ReplicaGrouping from a provided VariableSettings.

        The inverse of _to_variable_settings.

        Args:
            ir (Ir): The ir to use to create the ReplicaGrouping.
            variable_settings (VariableSettings): The PopART VariableSettings object
                to use to generate the replica_grouping with.

        Raises:
            ValueError: If an invalid VariableSettings object is passed.

        Returns:
            ReplicaGrouping: The ReplicaGrouping with the corresponding paramenters from the
                VariableSettings.
        """
        comm_group = variable_settings.getSharedVariableDomain()
        replicas = ir.replication_factor

        if comm_group.type == _ir.CommGroupType.Ungrouped and comm_group.replicaGroupSize == 0:
            self = ReplicaGrouping._from_params(ir, group_size=1)
        elif comm_group.type == _ir.CommGroupType.All:
            self = ReplicaGrouping._from_params(ir, group_size=replicas)
        elif comm_group.type == _ir.CommGroupType.Consecutive and replicas % comm_group.replicaGroupSize == 0:
            self = ReplicaGrouping._from_params(
                ir, group_size=comm_group.replicaGroupSize)
        elif comm_group.type == _ir.CommGroupType.Orthogonal:
            self = ReplicaGrouping._from_params(
                ir,
                group_size=comm_group.replicaGroupSize,
                stride=replicas // comm_group.replicaGroupSize)
        else:
            raise ValueError(
                f"VariableSettings with num_replicas={replicas}, "
                f"CommGroupType={comm_group.type } and replicaGroupSize"
                f"={comm_group.replicaGroupSize} is not currently supported")

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
        return f"ReplicaGrouping(num_replicas={self._ir.replication_factor}, " \
            f"stride={self.stride}, group_size={self.group_size})"

    def _to_variable_settings(
            self,
            retrieval_mode: Literal["one_per_group", "all_replicas"] = None
    ) -> VariableSettings:
        """Create a popart.VariableSettings object from this ReplicaGrouping's stride and group_size.

        Also takes a retrieval_mode due to the underlying construction of the VaribleSettings object.

        Args:
            retrieval_mode (Literal["one_per_group", "all_replicas"], optional):
                One of:
                - "one_per_group": Return only the first replica's variable per group.
                - "all_replicas": Return all replica's variables in every group.
                Defaults to None.

        Raises:
            ValueError: If incompatible stride and group_size arguments are passed.
            ValueError: If an invalid retrieval_mode is provided.

        Returns:
            VariableSettings: The popart equivalent of ReplicaGroupings.
        """

        replicas: int = self._ir.replication_factor
        comm_group: _ir.CommGroup
        if retrieval_mode is None or retrieval_mode == "one_per_group":
            var_ret_mode = VariableRetrievalMode.OnePerGroup
        elif retrieval_mode == "all_replicas":
            var_ret_mode = VariableRetrievalMode.AllReplicas
        else:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}")

        # If replica_grouping.group_size==1 => use CommGroupType::None
        if self.group_size == 1:
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.Ungrouped,
                                       replicaGroupSize=0)
        # If replica_grouping.group_size==N => use CommGroupType::All
        elif self.group_size == replicas:
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.All,
                                       replicaGroupSize=0)
        # If replica_grouping.stride==1 (and replica_grouping.group_size divides N) =>
        # use CommGroupType::Consecutive with size=replica_grouping.group_size
        elif self.stride == 1 and (replicas % self.group_size == 0):
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.Consecutive,
                                       replicaGroupSize=self.group_size)
        # If replica_grouping.stride==N/replica_grouping.group_size =>
        # use CommGroupType::Orthogonal with size=replica_grouping.group_size
        elif (replicas % self.group_size == 0) and self.stride == (
                replicas // self.group_size):
            comm_group = _ir.CommGroup(type=_ir.CommGroupType.Orthogonal,
                                       replicaGroupSize=self.group_size)
        else:
            raise ValueError(
                f"Replica grouping with num_replicas={replicas}, "
                f"stride={self.stride} and group_size={self.group_size} is not currently supported"
            )
        variable_settings = VariableSettings(comm_group, var_ret_mode)

        variable_settings.verify()
        return variable_settings
