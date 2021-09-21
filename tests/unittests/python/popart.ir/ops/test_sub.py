# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestSub:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()
        
        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = ops.sub(a, b)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("Sub", _ir.op.SubtractOp, g)

    def test_dunder(self):
        ir = pir.Ir()
        g = ir.main_graph()
        
        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = a - b
        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 2
        assert contains_op_of_type("Sub", _ir.op.SubtractOp, g)

    def test_ensure_tensor(self):
        ir = pir.Ir()
        g = ir.main_graph()
        
        with g:
            a = pir.variable(1)
            c = a - 2
        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 1
        assert len(ir.main_graph().get_constants()) == 1
        assert contains_op_of_type("Sub", _ir.op.SubtractOp, g)
