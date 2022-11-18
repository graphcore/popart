# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Iterable, Optional, List, Tuple, Set
from typing_extensions import Literal
from itertools import count, groupby, repeat, chain
import numpy as np
from collections import OrderedDict
import math
from functools import reduce

import popart._internal.ir as pb_ir
from popart import VariableSettings, VariableRetrievalMode, popart_exception

if TYPE_CHECKING:
    from popxl.ir import Ir


def _greatest_common_divisor(l: Iterable[int]):
    """Find the greatest common divisor for a list of ints."""
    return reduce(math.gcd, l)


def _get_order_of_first_appearance(l: Iterable[int]):
    """Obtain unique returned in order of appearance."""
    return np.array(list(OrderedDict.fromkeys(l)))


class ReplicaGrouping:
    def __init__(self):
        """Not intended to be called directly, use :py:func:`~popxl.Ir.replica_grouping` instead."""
        self._ir: "Ir"
        self._stride: Optional[int]  # The stride of the RG, if it has a constant stride
        self._group_size: int  # The group size for all groups
        self._num_groups: int  # The number of groups
        self._num_replicas: int  # Number of replicas
        self._assignment: List[int]  # List of assignments
        # A RG with const-stride that self maps to. If stride of self is already constant then this points to self.
        # If self is non-const-stride then self.const_rg is a RG that breaks the groups of self up until it has const-stride.
        # For example, if:
        # self.assignment           == [0, 0, 1, 1, 0, 0, 1, 1]
        # Then:
        # self.const_rg.assignment  == [0, 0, 1, 1, 2, 2, 3, 3]
        # self.to_device_map        == [0, 1, 0, 1]
        # self.to_host_map          == [0, 1]
        self.const_rg: ReplicaGrouping
        self.to_device_map: Optional[
            List[int]
        ]  # Maps the groups of `self` to each group of `const_rg`
        self.to_host_map: Optional[
            List[int]
        ]  # Maps the groups of `const_rg` to each group of `self`
        raise RuntimeError(
            "popxl.ReplicaGrouping cannot be constructed directly (use `ir.replica_grouping`)."
        )

    @classmethod
    def _from_params(
        cls, ir: "Ir", stride: int = 1, group_size: Optional[int] = None
    ) -> "ReplicaGrouping":
        num_replicas = ir.replication_factor

        if group_size is None:
            group_size = num_replicas // stride

        # Validate
        if not (0 < stride <= num_replicas):
            raise ValueError(
                "Stride must be greater than zero and less than or equal to number of replicas. "
                f"Stride: {stride}. Replicas: {num_replicas}"
            )
        if not (num_replicas / stride).is_integer():
            raise ValueError(
                "Stride must be a factor of the number of replicas. "
                f"Stride: {stride}. Replicas: {num_replicas}"
            )
        if not (0 < group_size <= num_replicas):
            raise ValueError(
                f"Group size must be greater than zero and less than or equal to number of replicas. "
                f"Group size: {group_size}. Replicas: {num_replicas}"
            )

        # Stride has no effect if group_size == 1
        stride = stride if group_size > 1 else 1

        # init
        self = super().__new__(cls)
        self._ir = ir
        self._stride = stride
        self._group_size = group_size
        self._num_groups = num_replicas // group_size
        self._num_replicas = num_replicas
        self._assignment = self._get_assignment(
            num_replicas, stride, group_size, self._num_groups
        )
        self.const_rg = self
        self.to_device_map = None
        self.to_host_map = None

        return self

    @classmethod
    def _from_assignment(cls, ir: "Ir", assignment: List[int]):
        num_replicas = ir.replication_factor

        # Validate
        unique, counts = np.unique(assignment, return_counts=True)
        if not len(set(counts)) == 1:
            raise ValueError(
                f"All groups should have the same size but they dont. (Group, Sizes): {list(zip(unique, counts))}"
            )

        if not np.all(unique == np.arange(len(unique))):
            raise ValueError(
                f"The group numbers are not numbered consecutively and/or do not start at 0. Groups: {unique}"
            )

        order_of_appearance = _get_order_of_first_appearance(assignment)
        if not np.all(order_of_appearance == unique):
            raise ValueError(
                f"When ordered by first appearance, the group numbers are not monotonically increasing. Ordering: {order_of_appearance}"
            )

        if len(assignment) != num_replicas:
            raise ValueError(
                f"The length of `assignment` ({len(assignment)}) does not equal the number of replicas ({num_replicas})."
            )

        # Determine if it is a const ReplicaGrouping
        strides = ReplicaGrouping._get_strides(assignment)
        if len(strides) == 1:
            return ir.replica_grouping(stride=strides.pop(), group_size=counts[0])

        # init
        self = super().__new__(cls)
        self._assignment = list(assignment)  # Copy
        self._group_size = counts[0]
        self._num_groups = len(counts)
        self._num_replicas = num_replicas
        self._ir = ir
        self._stride = None
        self.const_rg = ReplicaGrouping._to_const_rg(ir, assignment)
        self.to_device_map, self.to_host_map = ReplicaGrouping._get_maps(
            self.const_rg.assignment, self._assignment
        )

        return self

    @staticmethod
    def _get_assignment(
        num_replicas: int, stride: int, group_size: int, num_groups: int
    ) -> List[int]:
        """For a RG with constant stride, generate the assignment."""
        assign = [None] * num_replicas
        offset = 0
        for i in range(num_groups):
            while assign[offset] is not None:
                offset += 1
            assign[offset : offset + group_size * stride : stride] = [i] * group_size

        return assign

    @staticmethod
    def _get_strides(assignment: List[int]) -> Tuple[Set, List[Tuple[int, int]]]:
        """Return the set of strides from all groups.

        The stride is the distance between two consecutive replicas of the same group.

        For example if assignment == [0, 0, 1, 1, 0, 0, 1, 1]. The strides for group 0 are
        [1, 3, 1] and of group 1 are [1, 3, 1]. Therefore the set of strides are [1, 3].
        """
        # The result set.
        strides = set()
        # Groups are the unique numbers in assignments.
        groups = set(assignment)

        for group in groups:
            # For each group, list the indices of each instance of that group.
            group_indices = [index for index, g in enumerate(assignment) if g == group]
            if len(group_indices) > 1:
                # Work out the distance between the indices.
                group_strides = [
                    i - j for i, j in zip(group_indices[1:], group_indices[:-1])
                ]
                # Add distances to result.
                strides.update(group_strides)

        return strides

    @staticmethod
    def _to_const_rg(ir: "Ir", assignment: List[int]) -> "ReplicaGrouping":
        """Obtain a new replica grouping (const_rg) with a constant stride.

        Which has:
        A) A one to many mapping from self to const_rg
        B) The minimum number of groups possible"""
        # Determine current strides. New stride is the minimum
        strides = ReplicaGrouping._get_strides(assignment)
        new_stride = min(strides)

        # Split groups if >new_stride
        groups = set(assignment)
        assignment = list(assignment)  # Copy
        max_group_idx = max(assignment)
        for group in groups:
            # For each group, list the indices of each instance of that group.
            group_indices = [index for index, g in enumerate(assignment) if g == group]
            if len(group_indices) > 1:
                change_group = False
                for j in range(1, len(group_indices)):
                    stride = group_indices[j] - group_indices[j - 1]
                    # For each member of group, calculate the stride
                    # if stride>new_stride then make all remaining
                    # items in that group part of a new group. The new groups
                    # index is max_group_idx+1
                    if stride > new_stride:
                        change_group = True
                        max_group_idx += 1
                    if change_group:
                        # Change inplace the group
                        assignment[group_indices[j]] = max_group_idx

        # Calculate group sizes
        _, group_sizes = np.unique(assignment, return_counts=True)
        group_sizes = set(group_sizes)

        # New group size is greatest common divisor of all group sizes
        new_group_size = _greatest_common_divisor(group_sizes)
        new_stride = new_stride if new_group_size > 1 else 1

        const_rg = ir.replica_grouping(stride=new_stride, group_size=new_group_size)

        return const_rg

    @staticmethod
    def _get_maps(const_assignment, assignment) -> Tuple[List[int], List[int]]:
        """If has a non-const stride, obtain the to_device_map and to_host_map maps.

        to_device_map: Maps the groups of `self` to each group of `const_rg`
        to_host_map: Maps the groups of `const_rg` to each group of `self`

        For example, if:
        self.assignment           == [0, 0, 1, 1, 0, 0, 1, 1]
        Then:
        self.const_rg.assignment  == [0, 0, 1, 1, 2, 2, 3, 3]
        self.to_device_map        == [0, 1, 0, 1]
        self.to_host_map          == [0, 1]
        """
        to_device_map = []
        to_host_map = []
        seen_dynamic = set()
        seen_const = set()
        for const, dynamic in zip(const_assignment, assignment):
            if const not in seen_const:
                seen_const.add(const)
                to_device_map += [dynamic]
            if dynamic not in seen_dynamic:
                seen_dynamic.add(dynamic)
                to_host_map += [const]
        return to_device_map, to_host_map

    @property
    def _pb_replica_grouping(self) -> pb_ir.ReplicaGrouping:
        """Obtain Python Binding of Popart C++ ReplicaGrouping."""
        if self.is_const:
            try:
                return pb_ir.ReplicaGrouping(
                    self._num_replicas, self.stride, self.group_size
                )
            except popart_exception as e:
                raise ValueError(e)
        raise NotImplementedError(
            "This ReplicaGrouping has a non-constant stride and has no PopART C++ representation. "
            "You can't use this ReplicaGrouping with GCL ops. "
        )

    @property
    def stride(self) -> int:
        """
        Get the stride.

        Returns:
            int: The offset between elements in a replica group.
        Raises:
            NotImplementedError: if non-constant stride
        """
        if self.is_const:
            return self._stride
        else:
            raise NotImplementedError(
                "This ReplicaGrouping has a non-constant stride and has no PopART C++ representation. "
                "You can't use this ReplicaGrouping with GCL ops. "
            )

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
        return self._num_groups

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
        return self._assignment

    @property
    def is_const(self) -> bool:
        """Return True if has a constant stride and group size, and therefore can be represented using a PopART C++ ReplicaGrouping."""
        return self._stride is not None

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

        A good way to visualise the transpose is by considering the group matrix. For example,
        for the assignment `[0, 1, 0, 1, 2, 3, 2, 3]` the group matrix is:
        [[0, 2],
         [1, 3],
         [4, 6],
         [5, 7]]
        Whereby the first axis is the group index and the values are the replica index.
        The transpose of this matrix is:
        [[0, 1, 4, 5],
          2, 3, 6, 7]]
        Which converts back to the assignments `[0, 0, 1, 1, 0, 0, 1, 1]`.

        Returns:
            ReplicaGrouping: A "transpose" replica grouping of self.
        """
        if not self.is_const or (
            self.stride > 1 and self.stride * self.group_size != self._num_replicas
        ):
            # Create the group matrix from assignments.
            # The group matrix outer-most dimension is the group index and the values are replica index
            # e.g. if assignment = [0, 1, 0, 1, 2, 3, 2, 3], group_size = 2
            # group_matrix = [
            #   [0, 2],
            #   [1, 3],
            #   [4, 6],
            #   [5, 7],
            # ]
            group_matrix = sorted(zip(self.assignment, count()))
            group_matrix = [
                list(zip(*group))[1]
                for _, group in groupby(group_matrix, key=lambda x: x[0])
            ]
            # Transpose the group matrix
            # group_matrix_T = [
            #   [0, 1, 4, 5],
            #   [2, 3, 6, 7],
            # ]
            group_matrix_T = list(zip(*group_matrix))
            # Create the assignment list from the transposed group matrix
            # e.g. assignments_T = [0, 0, 1, 1, 0, 0, 1, 1]
            assignments_T = list(
                chain(
                    *[
                        zip(repeat(group_idx), group)
                        for group_idx, group in enumerate(group_matrix_T)
                    ]
                )
            )
            assignments_T = [
                group_idx for group_idx, _ in sorted(assignments_T, key=lambda x: x[1])
            ]
            return self._ir.replica_grouping_from_assignments(assignment=assignments_T)
        else:
            group_size = self.num_groups
            stride = 1 if self.stride > 1 else self.group_size
            return self._ir.replica_grouping(stride, group_size)

    def __repr__(self) -> str:
        """
        Return a string representation.

        Returns:
            str: A string representation of this ReplicaGrouping instance.
        """
        if self.is_const:
            return (
                f"ReplicaGrouping("
                f"num_replicas={self._num_replicas}, "
                f"stride={self.stride}, "
                f"group_size={self.group_size}, "
                f"num_groups={self.num_groups})"
            )
        else:
            return f"ReplicaGrouping(assignment={self.assignment})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ReplicaGrouping):
            raise TypeError(
                f"Value must be of type popxl.ReplicaGrouping. Type: {type(other)}. Value: {other}."
            )
        return self._ir == other._ir and self.assignment == other.assignment

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
        if not self.is_const:
            NotImplementedError(
                "This ReplicaGrouping has a non-constant stride and has no PopART C++ ReplicaGrouping or VariableSettings representation. "
            )

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
