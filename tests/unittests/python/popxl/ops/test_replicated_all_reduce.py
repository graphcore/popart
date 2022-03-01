# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestReplicatedAllReduce:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable(np.random.rand(3, 5, 7))
            c = ops.collectives.replicated_all_reduce(t)

        assert c.shape == t.shape
        assert len(g.tensors) == 2
        assert contains_op_of_type("ReplicatedAllReduce",
                                   _ir.op.collectives.ReplicatedAllReduceOp, g)

    def test_fn_inplace(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable(np.random.rand(3, 5, 7))
            c = ops.collectives.replicated_all_reduce_(t)

        assert c.shape == t.shape
        assert len(g.tensors) == 2
        assert contains_op_of_type(
            "ReplicatedAllReduceInplace",
            _ir.op.collectives.ReplicatedAllReduceInplaceOp, g)
