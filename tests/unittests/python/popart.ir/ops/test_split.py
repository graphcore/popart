# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type
from popart import popart_exception
import pytest


class TestSplit:
    def test_fn_int(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable([[1, 2], [3, 4], [5, 6], [7, 8]])
            splits = 2
            c = ops.split(t, splits)

        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        contains_op_of_type("Split", _ir.op.SplitOp, g)

    def test_fn_list(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
            splits = [1, 3, 1]
            c = ops.split(t, splits)

        assert len(g.get_tensors()) == 4
        assert len(g.get_variables()) == 1
        contains_op_of_type("Split", _ir.op.SplitOp, g)

    def test_invalid_int(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
            splits = 2
            with pytest.raises(ValueError) as excinfo:
                c = ops.split(t, splits)

    def test_invalid_list(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
            splits = [1, 2]
            with pytest.raises(popart_exception) as excinfo:
                c = ops.split(t, splits)
