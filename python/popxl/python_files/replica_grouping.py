# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Optional, List
from typing_extensions import Literal

import popart._internal.ir as _ir
from popart import VariableSettings, VariableRetrievalMode

if TYPE_CHECKING:
    from popxl.ir import Ir


class ReplicaGrouping:
    def __init__(self):
        """Not intended to be called directly, use :py:func:`~popxl.Ir.replica_grouping` instead."""
        raise RuntimeError(
            "popxl.ReplicaGrouping cannot be constructed directly (use `ir.replica_grouping`)."
        )

    @classmethod
    def _from_params(
        cls, ir: "Ir", stride: int = 1, group_size: Optional[int] = None
    ) -> "ReplicaGrouping":
        self = super().__new__(cls)
        self._ir = ir
        self._stride = stride
        replicas = self._ir.replication_factor

        if group_size is None:
            self._group_size = replicas // stride
        else:
            self._group_size = group_size

        if stride > replicas:
            raise ValueError(
                f"Stride ({stride}) cannot be large then the number of replicas ({replicas})"
            )
        if not (replicas / stride).is_integer():
            raise ValueError(
                f"Stride ({stride}) must be a factor of the number of replicas ({replicas})"
            )
        if self._group_size > replicas:
            raise ValueError(
                f"Group size ({self._group_size}) cannot be large then the number of replicas ({replicas})"
            )

        return self

    @classmethod
    def _from_variable_settings(
        cls, ir: "Ir", variable_settings: VariableSettings
    ) -> "ReplicaGrouping":
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

        if (
            comm_group.type == _ir.CommGroupType.Ungrouped
            and comm_group.replicaGroupSize == 0
        ):
            self = ReplicaGrouping._from_params(ir, group_size=1)
        elif comm_group.type == _ir.CommGroupType.All:
            self = ReplicaGrouping._from_params(ir, group_size=replicas)
        elif (
            comm_group.type == _ir.CommGroupType.Consecutive
            and replicas % comm_group.replicaGroupSize == 0
        ):
            self = ReplicaGrouping._from_params(
                ir, group_size=comm_group.replicaGroupSize
            )
        elif comm_group.type == _ir.CommGroupType.Orthogonal:
            self = ReplicaGrouping._from_params(
                ir,
                group_size=replicas // comm_group.replicaGroupSize,
                stride=comm_group.replicaGroupSize,
            )
        else:
            raise ValueError(
                f"VariableSettings with num_replicas={replicas}, "
                f"CommGroupType={comm_group.type } and replicaGroupSize"
                f"={comm_group.replicaGroupSize} is not currently supported"
            )

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
        replicas = self._ir.replication_factor
        if self.group_size > 1:
            return replicas // self._group_size
        return replicas

    @property
    def assignment(self) -> List[int]:
        """Obtain the group each replica is assigned to.

        Examples (with `ir.replication_factor = 8`):

        .. code-block:: python

            ir.replica_grouping(stride=1, group_size=8).assignment
            [0, 0, 0, 0, 0, 0, 0, 0]

            ir.replica_grouping(stride=1, group_size=1).assignment
            [0, 1, 2, 3, 4, 5, 6, 7]

            ir.replica_grouping(stride=1, group_size=2).assignment
            [0, 0, 1, 1, 2, 2, 3, 3]

            ir.replica_grouping(stride=2, group_size=2).assignment
            [0, 1, 0, 1, 2, 3, 2, 3]

            ir.replica_grouping(stride=1, group_size=4).assignment
            [0, 0, 0, 0, 1, 1, 1, 1]

            ir.replica_grouping(stride=2, group_size=4).assignment
            [0, 1, 0, 1, 0, 1, 0, 1]

        Returns:
            List[int]: A list where the index is the replica and value is the group index
        """
        assign = [None] * self._ir.replication_factor
        offset = 0
        for i in range(self.num_groups):
            while assign[offset] is not None:
                offset += 1
            assign[offset : offset + self.group_size * self.stride : self.stride] = [
                i
            ] * self.group_size

        return assign

    def transpose(self) -> "ReplicaGrouping":
        """Return the transpose of this replica grouping.

        A replica grouping whereby the first element of each group is the first new group, the
        second element of each group is the second group etc.

        Examples:

        .. code-block:: text

            [0, 0, 0, 0] -> [0, 1, 2, 3]
            [0, 1, 0, 1] -> [0, 0, 1, 1]
            [0, 0, 1, 1] -> [0, 1, 0, 1]
            [0, 1, 2, 3] -> [0, 0, 0, 0]

        Some transposes cannot be represented with just a stride and group size and therefore
        cannot be created. For example for `num_replicas=8`, `stride=2` and `group_size=2`, the assignments
        are `[0, 1, 0, 1, 2, 3, 2, 3]` and the transpose is `[0, 0, 1, 1, 0, 0, 1, 1]`.

        Raises:
            ValueError: If the transpose cannot be represented with a replica grouping.

        Returns:
            ReplicaGrouping: A "transpose" replica grouping of self.
        """
        if (
            self.stride > 1
            and self.stride * self.group_size != self._ir.replication_factor
        ):
            raise ValueError(
                "The transpose of this replica grouping cannot be represented with a replica grouping."
            )

        group_size = self.num_groups
        stride = 1 if self.stride > 1 else self.group_size
        return self._ir.replica_grouping(stride, group_size)

    def __repr__(self) -> str:
        """
        Return a string representation.

        Returns:
            str: A string representation of this ReplicaGrouping instance.
        """
        return (
            f"ReplicaGrouping(num_replicas={self._ir.replication_factor}, "
            f"stride={self.stride}, group_size={self.group_size}, num_groups={self.num_groups})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ReplicaGrouping):
            raise TypeError(
                f"Value must be of type popxl.ReplicaGrouping. Type: {type(other)}. Value: {other}."
            )
        return (
            self.stride == other.stride
            and self.group_size == other.group_size
            and self._ir == other._ir
        )

    def _to_variable_settings(
        self, retrieval_mode: Literal["one_per_group", "all_replicas"] = None
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
            VariableSettings: The PopART equivalent of ReplicaGroupings.
        """
        comm_group = self._to_comm_group()
        if retrieval_mode is None or retrieval_mode == "one_per_group":
            var_ret_mode = VariableRetrievalMode.OnePerGroup
        elif retrieval_mode == "all_replicas":
            var_ret_mode = VariableRetrievalMode.AllReplicas
        else:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}")
        variable_settings = VariableSettings(comm_group, var_ret_mode)
        variable_settings.verify()
        return variable_settings

    def _to_comm_group(self) -> _ir.CommGroup:
        replicas = self._ir.replication_factor

        if self.group_size == 1:
            return _ir.CommGroup(type=_ir.CommGroupType.Ungrouped, replicaGroupSize=0)
        # If replica_grouping.group_size==N => use CommGroupType::All
        elif self.group_size == replicas:
            return _ir.CommGroup(type=_ir.CommGroupType.All, replicaGroupSize=0)
        # If replica_grouping.stride==1 (and replica_grouping.group_size divides N) =>
        # use CommGroupType::Consecutive with size=replica_grouping.group_size
        elif self.stride == 1 and (replicas % self.group_size == 0):
            return _ir.CommGroup(
                type=_ir.CommGroupType.Consecutive, replicaGroupSize=self.group_size
            )
        # If replica_grouping.stride==N/replica_grouping.group_size =>
        # use CommGroupType::Orthogonal with size=replica_grouping.stride
        elif (replicas % self.group_size == 0) and self.stride == (
            replicas // self.group_size
        ):
            return _ir.CommGroup(
                type=_ir.CommGroupType.Orthogonal, replicaGroupSize=self.stride
            )
        raise ValueError(
            f"Replica grouping with num_replicas={replicas}, "
            f"stride={self.stride} and group_size={self.group_size} is not currently supported"
        )
