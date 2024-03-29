# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestReplicatedReduceScatter:
    def test_fn(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = 2
        g = ir.main_graph

        with g:
            t = popxl.variable(np.random.rand(2, 5, 7))
            scattered_t = ops.collectives.replicated_reduce_scatter(t)

        assert t.nelms == scattered_t.nelms * 2
        assert len(g.tensors) == 2
        assert contains_op_of_type(
            "ReplicatedReduceScatter", _ir.op.collectives.ReplicatedReduceScatterOp, g
        )
