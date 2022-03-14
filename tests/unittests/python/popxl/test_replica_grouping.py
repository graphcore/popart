# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import popxl


class TestReplicaGrouping:
    def test_replica_grouping_constructor(self):
        """ Test that calling the constructor raises an error. """
        with pytest.raises(RuntimeError):
            popxl.ReplicaGrouping()

    @pytest.mark.parametrize(
        'num_replicas, stride, group_size',
        [
            (8, 1, 8),  # 00000000
            (8, 1, 2),  # 00112233
            (8, 2, 4),  # 01010101
            (8, 1, 4)  # 00001111
        ])
    def test_replica_grouping_construct(self, num_replicas, stride,
                                        group_size):
        """ Test constructing some replica grouping objects. """
        ir = popxl.Ir()
        ir.replication_factor = num_replicas
        rg = ir.replica_grouping(stride=stride, group_size=group_size)
        assert rg.stride == stride
        assert rg.group_size == group_size

    def test_replica_grouping_repr(self):
        ir = popxl.Ir()
        ir.replication_factor = 8
        rg = ir.replica_grouping(stride=2, group_size=4)
        assert repr(
            rg) == 'ReplicaGrouping(num_replicas=8, stride=2, group_size=4)'
