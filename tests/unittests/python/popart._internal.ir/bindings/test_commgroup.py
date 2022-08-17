# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir


class TestCommGroup:
    def test_init(self):
        _ir.CommGroup(_ir.ReplicaGrouping(8))

    def test_to_replica_grouping(self):
        _ir.CommGroup().toReplicaGrouping(8)
