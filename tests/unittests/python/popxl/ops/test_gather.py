# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from popxl import dtypes
from utils import contains_op_of_type


class TestGather:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable([[1, 2], [3, 4]])
            indices = popxl.variable([[0, 1], [1, 0]], dtype=dtypes.int32)
            _ = ops.gather(t, indices)

        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Gather", _ir.op.GatherOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable([[1, 2], [3, 4]])
            indices = popxl.variable([[0, 1], [1, 0]], dtype=dtypes.int32)
            _ = t[indices]

        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 2
        assert contains_op_of_type("Gather", _ir.op.GatherOp, g)


class TestTiedGather:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable([[1, 2], [3, 4]])
            indices = popxl.variable([[0, 1], [1, 0]])
            _ = ops.tied_gather(t, indices)

        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("PopartTiedGather", _ir.op.TiedGatherOp, g)
