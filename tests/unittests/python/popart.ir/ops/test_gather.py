# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestGather:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable([[1, 2], [3, 4]])
            indices = pir.variable([[0, 1], [1, 0]])
            c = ops.gather(t, indices)

        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 2
        contains_op_of_type("Gather", _ir.op.GatherOp, g)


class TestTiedGather:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable([[1, 2], [3, 4]])
            indices = pir.variable([[0, 1], [1, 0]])
            c = ops.tied_gather(t, indices)

        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 2
        contains_op_of_type("PopartTiedGather", _ir.op.TiedGatherOp, g)
