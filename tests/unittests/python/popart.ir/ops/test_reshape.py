# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import pytest
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestReshape:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = ops.reshape(a, (3, 2, 1))
        assert c.shape == (3, 2, 1)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Reshape", _ir.op.ReshapeOp, g)

    def test_tensor_method(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = a.reshape((2, 3, 1))
        assert c.shape == (2, 3, 1)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Reshape", _ir.op.ReshapeOp, g)

    def test_negative(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = a.reshape((-1, 1))
        assert c.shape == (6, 1)
        assert contains_op_of_type("Reshape", _ir.op.ReshapeOp, g)

    @pytest.mark.parametrize("shape", [(-1, 0, 1), (6, -2, 1)])
    def test_invalid_value(self, shape):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            with pytest.raises(ValueError) as excinfo:
                c = a.reshape(shape)
            message = str(excinfo.value)
            assert "Invalid shape value in reshape" in message
            assert str(shape) in message

    def test_double_negative_one(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            with pytest.raises(ValueError) as excinfo:
                c = a.reshape((-1, -1, 1))
            message = str(excinfo.value)
            assert "Reshape shape can contain at most one '-1' value" in message
            assert "(-1, -1, 1)" in message
