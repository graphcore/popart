# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from popart.ir import dtypes
from utils import contains_op_of_type


class TestScatter:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            t = pir.variable([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ])
            indices = pir.variable([
                [1, 0, 2],
                [0, 2, 1],
            ],
                                   dtype=dtypes.int32)
            values = pir.variable([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ])
            c = ops.scatter(t, indices, values)

        assert len(g.tensors) == 4
        assert len(g.variables) == 3
        assert contains_op_of_type("Scatter", _ir.op.ScatterOp, g)
