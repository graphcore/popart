# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestPrintTensor:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = ops.print_tensor(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("PrintTensor", _ir.op.PrintTensorOp, g)
