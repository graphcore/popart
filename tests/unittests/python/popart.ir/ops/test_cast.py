# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestCast:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(1, pir.float32)
            b = ops.cast(a, pir.float16)

        assert b.dtype == pir.float16
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Cast", _ir.op.CastOp, g)
