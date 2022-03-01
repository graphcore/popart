# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import pytest
import numpy as np
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


@pytest.mark.parametrize("inplace", [False, True])
class TestTranspose:
    def check_contains_transpose(self, g: popxl.Graph,
                                 inplace: bool = False) -> bool:
        if inplace:
            return contains_op_of_type("TransposeInplace",
                                       _ir.op.TransposeInplaceOp, g)
        return contains_op_of_type("Transpose", _ir.op.TransposeOp, g)

    def test_fn(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            if inplace:
                c = ops.transpose_(a, (0, 2, 1))
            else:
                c = ops.transpose(a, (0, 2, 1))
        assert c.shape == (1, 3, 2)
        assert len(g.tensors) == 2
        print(g._pb_graph.getOps())
        assert self.check_contains_transpose(g, inplace)

    def test_tensor_method(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            if inplace:
                c = a.transpose_()
            else:
                c = a.transpose()
        assert c.shape == (3, 2, 1)
        assert len(g.tensors) == 2
        assert self.check_contains_transpose(g, inplace)

    def test_numpy_transpose(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            if inplace:
                c = a.T_
            else:
                c = a.T
        assert c.shape == (3, 2, 1)
        assert len(g.tensors) == 2
        assert self.check_contains_transpose(g, inplace)

    def test_out_of_range(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            with pytest.raises(ValueError) as excinfo:
                if inplace:
                    c = ops.transpose_(a, (5, 4, 3))
                else:
                    c = ops.transpose(a, (5, 4, 3))
            message = str(excinfo.value)
            assert "Values in permutation must be less than" in message
            assert "rank 3" in message
            assert "(5, 4, 3)" in message
