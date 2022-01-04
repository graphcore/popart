# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops
import numpy as np
import pytest
from popart.ir import dtypes
import popart._internal.ir as _ir
from utils import contains_op_of_type


@pytest.mark.parametrize("available_memory_proportion", (0.15, 1.0, None))
@pytest.mark.parametrize(
    "serialise_mode",
    (ops.SerialiseMode.NoSerialisation, ops.SerialiseMode.ReducingDim,
     ops.SerialiseMode.InputChannels, ops.SerialiseMode.OutputChannels))
@pytest.mark.parametrize("serialise_factor", (1, 4))
@pytest.mark.parametrize("output_type", (dtypes.float32, dtypes.float16, None))
@pytest.mark.parametrize("partials_type", (dtypes.float32, dtypes.float16))
class TestMatMul:
    def test_fn(self, available_memory_proportion, serialise_mode,
                serialise_factor, output_type, partials_type):
        ir = pir.Ir()
        with ir.main_graph():
            a = pir.variable(np.random.rand(4, 4))
            b = pir.variable(np.random.rand(4, 4))
            c = ops.matmul(a, b, available_memory_proportion, serialise_mode,
                           serialise_factor, output_type, partials_type)
        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 2
        assert contains_op_of_type("MatMul", _ir.op.MatMulOp, ir.main_graph())

    def test_dunder(self, available_memory_proportion, serialise_mode,
                    serialise_factor, output_type, partials_type):
        ir = pir.Ir()
        del available_memory_proportion, serialise_mode, serialise_factor, output_type, partials_type
        with ir.main_graph():
            a = pir.variable(np.random.rand(4, 4))
            b = pir.variable(np.random.rand(4, 4))
            c = a @ b
        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 2
        assert contains_op_of_type("MatMul", _ir.op.MatMulOp, ir.main_graph())
