# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type
from popxl.utils import host_pow2scale_cast_to_fp8
import pytest
import numpy as np


class TestCast:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1, popxl.float32, name="t0")
            b = ops.cast(a, popxl.float16)

        assert b.dtype == popxl.float16
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Cast", _ir.op.CastOp, g)
        assert b.id.count(a.id) == 1

    @pytest.mark.parametrize("dtype", [popxl.float8_143, popxl.float8_152])
    def test_pow2scale_cast_to_fp8(self, dtype):
        """Cast fp32 to fp8"""
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1, popxl.float16, name="t0")
            log2_scale = popxl.constant(0, popxl.int32, name="log2_scale")
            b = ops.pow2scale_cast_to_fp8(a, log2_scale, dtype)

        assert b.dtype == dtype
        assert len(g.tensors) == 3  # in, log2_scale, out
        assert len(g.variables) == 1
        assert contains_op_of_type("Pow2ScaleThenCast", _ir.op.Pow2ScaleThenCastOp, g)
        assert b.id.count(a.id) == 1

    @pytest.mark.parametrize("from_dtype", [popxl.float8_143, popxl.float8_152])
    @pytest.mark.parametrize("to_dtype", [popxl.float16, popxl.float32])
    def test_pow2scale_cast_from_fp8(self, from_dtype, to_dtype):
        """Cast fp8 to fp32/fp64"""
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            data = np.random.random([1, 2, 3]).astype(np.float32)
            input0 = popxl.h2d_stream(
                data.shape,
                from_dtype,
                name="in_stream_0",
            )
            data = host_pow2scale_cast_to_fp8(data, from_dtype, 0)
            a = ops.host_load(input0, "a")
            log2_scale = popxl.constant(0, popxl.int32, name="log2_scale")
            b = ops.pow2scale_cast_from_fp8(a, log2_scale, to_dtype)

        assert b.dtype == to_dtype
        print(g.tensors)
        assert len(g.tensors) == 5
        assert contains_op_of_type("CastThenPow2Scale", _ir.op.CastThenPow2ScaleOp, g)
        assert b.id.count(a.id) == 1
