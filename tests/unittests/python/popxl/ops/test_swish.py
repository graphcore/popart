# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestSwish:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = ops.swish(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Swish", _ir.op.SwishOp, g)

    def test_fn_inplace(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = ops.swish_(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("SwishInplace", _ir.op.SwishInplaceOp, g)
