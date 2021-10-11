# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import pytest
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestReshape:
    def test_squeeze_all(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((4, 1, 3, 1)))
            c = ops.squeeze(a)
        assert c.shape == (4, 3)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Reshape", _ir.op.ReshapeOp, g)

    def test_squeeze_specified(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((4, 1, 3, 1, 4, 5)))
            c = ops.squeeze(a, axes=[1, 3])
        assert c.shape == (4, 3, 4, 5)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Reshape", _ir.op.ReshapeOp, g)

    def test_squeeze_specified_negative(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((4, 1, 3, 1, 4, 5, 1)))
            c = ops.squeeze(a, axes=[-4, 1])
        assert c.shape == (4, 3, 4, 5, 1)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Reshape", _ir.op.ReshapeOp, g)

    def test_duplicate_axes(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            with pytest.raises(ValueError) as excinfo:
                c = ops.squeeze(a, axes=[2, 2])

    def test_axes_too_large(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            with pytest.raises(ValueError) as excinfo:
                c = ops.squeeze(a, axes=[3])

    def test_axes_not_squeezable(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            with pytest.raises(ValueError) as excinfo:
                c = ops.squeeze(a, axes=[1])
