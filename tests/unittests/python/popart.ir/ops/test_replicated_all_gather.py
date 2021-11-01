# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestReplicatedReduceScatter:
    def test_fn(self):
        ir = pir.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = 2
        g = ir.main_graph()

        with g:
            t = pir.variable(np.random.rand(2, 5, 7))
            gathered_t = ops.collectives.replicated_all_gather(t)

        assert t.nelms * 2 == gathered_t.nelms
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("ReplicatedAllGather",
                                   _ir.op.collectives.ReplicatedAllGatherOp, g)
