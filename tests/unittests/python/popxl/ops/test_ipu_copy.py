# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestIpuCopy:
    def test_fn_with_producer(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            with popxl.ipu(0):
                a_0 = a + 1
                ir._pb_ir.logIr()
            _ = ops.ipu_copy(a_0, 1)
        assert len(g.variables) == 1
        assert len(g.tensors) == 4
        assert contains_op_of_type("IpuCopy", _ir.op.IpuCopyOp, g)

    def test_fn_producer_without_ipu(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            # None will disable annotations of ipu. Not recommended
            # but ipu_copy should throw an error if it happens.
            with popxl.ipu(None):
                a_0 = a + 1
            with pytest.raises(ValueError) as excinfo:
                _ = ops.ipu_copy(a_0, 1)
            msg = str(excinfo.value)
            assert "Could not infer the ipu" in msg

    def test_fn_with_no_producer(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = ops.ipu_copy(a, 1, 0)
        assert len(g.variables) == 1
        assert len(g.tensors) == 2
        assert contains_op_of_type("IpuCopy", _ir.op.IpuCopyOp, g)

    def test_fn_with_no_producer_error(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            with pytest.raises(ValueError) as excinfo:
                _ = ops.ipu_copy(a, 1)
            msg = str(excinfo.value)
            assert "Could not infer the ipu" in msg
            assert "Please specify `source`" in msg

    def test_tensor_fns(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            a_1 = a.copy_to_ipu(1, 0)
            _ = a_1.copy_to_ipu(0)

            c = popxl.constant(2)
            c_1 = c.copy_to_ipu(1, 0)
            _ = c_1.copy_to_ipu(0)
