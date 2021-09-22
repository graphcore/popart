# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import pytest
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestTranspose:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = ops.transpose(a, (0, 2, 1))
        assert c.shape == (1, 3, 2)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Transpose", _ir.op.TransposeOp, g)

    def test_tensor_method(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = a.transpose()
        assert c.shape == (3, 2, 1)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Transpose", _ir.op.TransposeOp, g)

    def test_numpy_transpose(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = a.T
        assert c.shape == (3, 2, 1)
        assert len(g.get_tensors()) == 2
        assert contains_op_of_type("Transpose", _ir.op.TransposeOp, g)

    def test_out_of_range(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            with pytest.raises(ValueError) as excinfo:
                c = ops.transpose(a, (5, 4, 3))
            message = str(excinfo.value)
            assert "Values in permutation must be less than" in message
            assert "rank 3" in message
            assert "(5, 4, 3)" in message
