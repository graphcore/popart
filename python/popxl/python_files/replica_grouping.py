# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Optional, List
from typing_extensions import Literal

import popart._internal.ir as _ir
from popart import VariableSettings, VariableRetrievalMode, popart_exception

if TYPE_CHECKING:
    from popxl.ir import Ir


class ReplicaGrouping:
    def __init__(self):
        """Not intended to be called directly, use :py:func:`~popxl.Ir.replica_grouping` instead."""
        self._pb_replica_grouping: _ir.ReplicaGrouping = None
        raise RuntimeError(
            "popxl.ReplicaGrouping cannot be constructed directly (use `ir.replica_grouping`)."
        )

    @classmethod
    def _from_params(
        cls, ir: "Ir", stride: int = 1, group_size: Optional[int] = None
    ) -> "ReplicaGrouping":
        self = super().__new__(cls)
        num_replicas = ir.replication_factor
        group_size = num_replicas // stride if group_size is None else group_size
        try:
            self._pb_replica_grouping = _ir.ReplicaGrouping(
                num_replicas, stride, group_size
            )
        except popart_exception as e:
            raise ValueError(e)
        return self

    @classmethod
    def _from_pb_replica_grouping(
        cls, pb_replica_grouping: _ir.ReplicaGrouping
    ) -> "ReplicaGrouping":
        self = super().__new__(cls)
        self._pb_replica_grouping = pb_replica_grouping
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
        self = super().__new__(cls)
        num_replicas = ir.replication_factor
        try:
            self._pb_replica_grouping = variable_settings.getReplicaGrouping(
                num_replicas
            )
        except popart_exception as e:
            raise ValueError(e)
        return self

    @property
    def stride(self) -> int:
        """
        Get the stride.

        Returns:
            int: The offset between elements in a replica group.
        """
        return self._pb_replica_grouping.getStride()

    @property
    def group_size(self) -> int:
        """
        Get the group size.

        Returns:
            int: The number of replicas in each replica group.
        """
        return self._pb_replica_grouping.getGroupSize()

    @property
    def num_groups(self) -> int:
        """
        Get the number of groups.

        Returns:
            int: The number of replica groups.
        """
        return self._pb_replica_grouping.getNumGroups()

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
        return [
            self._pb_replica_grouping.getGroupAt(replica)
            for replica in range(self._pb_replica_grouping.getNumReplicas())
        ]

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
        try:
            return ReplicaGrouping._from_pb_replica_grouping(
                self._pb_replica_grouping.getTranspose()
            )
        except popart_exception as e:
            raise ValueError(e)

    def __repr__(self) -> str:
        """
        Return a string representation.

        Returns:
            str: A string representation of this ReplicaGrouping instance.
        """
        return (
            f"ReplicaGrouping("
            f"num_replicas={self._pb_replica_grouping.getNumReplicas()}, "
            f"stride={self.stride}, "
            f"group_size={self.group_size}, "
            f"num_groups={self.num_groups})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ReplicaGrouping):
            raise TypeError(
                f"Value must be of type popxl.ReplicaGrouping. Type: {type(other)}. Value: {other}."
            )
        return self._pb_replica_grouping == other._pb_replica_grouping

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
        if retrieval_mode is None or retrieval_mode == "one_per_group":
            var_ret_mode = VariableRetrievalMode.OnePerGroup
        elif retrieval_mode == "all_replicas":
            var_ret_mode = VariableRetrievalMode.AllReplicas
        else:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}")

        try:
            return VariableSettings(self.stride, self.group_size, var_ret_mode)
        except popart_exception as e:
            raise ValueError(e)
