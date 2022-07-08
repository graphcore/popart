# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import pytest
from popxl import dtypes
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestMatMul:
    @pytest.mark.parametrize("available_memory_proportion", (0.15, 1.0, None))
    @pytest.mark.parametrize("output_type", (dtypes.float32, dtypes.float16, None))
    @pytest.mark.parametrize("partials_type", (dtypes.float32, dtypes.float16))
    def test_fn(self, available_memory_proportion, output_type, partials_type):
        ir = popxl.Ir()
        with ir.main_graph:
            a = popxl.variable(np.random.rand(4, 4))
            b = popxl.variable(np.random.rand(4, 4))
            _ = ops.matmul(
                a, b, available_memory_proportion, output_type, partials_type
            )
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 2
        assert contains_op_of_type("MatMul", _ir.op.MatMulOp, ir.main_graph)

    def test_dunder(self):
        ir = popxl.Ir()
        with ir.main_graph:
            a = popxl.variable(np.random.rand(4, 4))
            b = popxl.variable(np.random.rand(4, 4))
            _ = a @ b
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 2
        assert contains_op_of_type("MatMul", _ir.op.MatMulOp, ir.main_graph)

    def test_ensure_tensor(self):
        ir = popxl.Ir()
        with ir.main_graph:
            a = popxl.variable(np.random.rand(4, 4))
            b = np.random.rand(4, 4)
            _ = a @ b
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 1
        assert len(ir.main_graph.constants) == 1
        assert contains_op_of_type("MatMul", _ir.op.MatMulOp, ir.main_graph)

    def test_ensure_tensor_lhs(self):
        ir = popxl.Ir()
        with ir.main_graph:
            a = np.random.rand(4, 4)
            b = popxl.variable(np.random.rand(4, 4))
            _ = a @ b
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 1
        assert len(ir.main_graph.constants) == 1
        assert contains_op_of_type("MatMul", _ir.op.MatMulOp, ir.main_graph)

    @pytest.mark.parametrize("partials_type", ("", "float", "half"))
    def test_partials_session_option(self, partials_type):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.partialsTypeMatMuls = partials_type
        with ir.main_graph:
            a = popxl.variable(np.random.rand(4, 4))
            b = popxl.variable(np.random.rand(4, 4))
            _ = ops.matmul(a, b)
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 2
        op = ir.main_graph._pb_graph.getOps()[0]
        assert isinstance(op, _ir.op.MatMulOp)
        _partials = (
            _ir.op.MatMulPartialsType.HALF
            if partials_type == "half"
            else _ir.op.MatMulPartialsType.FLOAT
        )
        assert op.getPartialsType() == _partials

    def test_partials_session_option_fail(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.partialsTypeMatMuls = "foo"
        with ir.main_graph:
            a = popxl.variable(np.random.rand(4, 4))
            b = popxl.variable(np.random.rand(4, 4))
            with pytest.raises(ValueError):
                _ = ops.matmul(a, b)
