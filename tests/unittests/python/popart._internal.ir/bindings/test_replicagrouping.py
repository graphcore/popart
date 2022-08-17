# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import itertools

import popart._internal.ir as _ir


class TestReplicaGrouping:
    def test_init(self):
        _ir.ReplicaGrouping(8, 2, 4)
        _ir.ReplicaGrouping(8)

    def test_get_num_replicas(self):
        grouping = _ir.ReplicaGrouping(8, 2, 4)
        assert grouping.getNumReplicas() == 8

    def test_get_stride(self):
        grouping = _ir.ReplicaGrouping(8, 2, 4)
        assert grouping.getStride() == 2

    def test_get_group_size(self):
        grouping = _ir.ReplicaGrouping(8, 2, 4)
        assert grouping.getGroupSize() == 4

    def test_get_num_groups(self):
        grouping = _ir.ReplicaGrouping(32, 2, 4)
        assert grouping.getNumGroups() == 8

    def test_get_group_at(self):
        grouping = _ir.ReplicaGrouping(8, 2, 4)
        result = [grouping.getGroupAt(replica) for replica in range(8)]
        assert result == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_get_index_in_group_at(self):
        grouping = _ir.ReplicaGrouping(8, 2, 4)
        result = [grouping.getIndexInGroupAt(replica) for replica in range(8)]
        assert result == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_get_replica_at(self):
        grouping = _ir.ReplicaGrouping(8, 1, 2)
        result = [
            grouping.getReplicaAt(group, index)
            for group, index in itertools.product(range(4), range(2))
        ]
        assert result == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_get_replicas_at(self):
        grouping = _ir.ReplicaGrouping(8, 1, 2)
        result = [grouping.getReplicasAt(group) for group in range(4)]
        assert result == [[0, 1], [2, 3], [4, 5], [6, 7]]

    def test_get_transpose(self):
        grouping = _ir.ReplicaGrouping(8, 2, 4)
        assert grouping.getTranspose() == _ir.ReplicaGrouping(8, 1, 2)

    def test_str(self):
        grouping = _ir.ReplicaGrouping(8, 2, 4)
        assert grouping.str() == "ReplicaGrouping(numReplicas=8, stride=2, groupSize=4)"

    def test___eq__(self):
        a = _ir.ReplicaGrouping(8, 2, 4)
        b = _ir.ReplicaGrouping(8, 2, 4)
        c = _ir.ReplicaGrouping(8, 2, 2)
        assert a == b
        assert not a == c

    def test___ne__(self):
        a = _ir.ReplicaGrouping(8, 2, 4)
        b = _ir.ReplicaGrouping(8, 2, 4)
        c = _ir.ReplicaGrouping(8, 2, 2)
        assert not a != b
        assert a != c
