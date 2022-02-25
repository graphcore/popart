# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestConcat:
    """Ensure concat API is available and raises exceptions when expected"""

    def test_fn_2(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable((1, ))
            b = pir.variable((2, ))
            c = ops.concat((a, b))
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Concat", _ir.op.ConcatOp, g)

    def test_fn_3(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable((1, ))
            b = pir.variable((2, ))
            c = pir.variable((3, ))
            d = ops.concat((a, b, c))
        assert len(g.tensors) == 4
        assert len(g.variables) == 3
        assert contains_op_of_type("Concat", _ir.op.ConcatOp, g)

    def test_fn_error(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(np.zeros((2, 4)))
            b = pir.variable(np.zeros((3, )))
            with pytest.raises(popart.popart_exception):
                c = ops.concat((a, b))
