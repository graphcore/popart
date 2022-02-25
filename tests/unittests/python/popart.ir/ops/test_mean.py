# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestMean:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = ops.mean(a, axis=[0, -1])
        assert len(g.tensors) == 2
        assert contains_op_of_type("ReduceMean", _ir.op.ReduceMeanOp, g)
