# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestScatterReduce:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        indices = [3, 0, 1]

        with g:
            x_t = popxl.variable(x)
            indices_t = popxl.variable(indices)
            _ = ops.scatter_reduce(
                data=x_t, indices=indices_t, axis=0, axis_size=5, reduction="sum"
            )

        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("ScatterReduce", _ir.op.ScatterReduceOp, g)
