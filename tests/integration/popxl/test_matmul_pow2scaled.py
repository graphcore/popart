# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

import popxl
import popxl.ops as ops

from popxl.utils import host_pow2scale_cast_to_fp8


@pytest.mark.parametrize("format_", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("log2_scale", [-1, 0, 1])
def test_fp8_matmul(format_, log2_scale):
    """Test that the `matmul_pow2scaled` op produces a float16 result when
    multiplying two float8 tensors that is approximately close to the same
    two tensors in float16 multiplied using NumPy.
    """
    shape = (2, 2)

    ir = popxl.Ir()
    main_graph = ir.main_graph
    x32_lhs: np.ndarray = np.random.rand(*shape)
    x32_rhs: np.ndarray = np.random.rand(*shape)

    # The result to compare the output of the op to.
    # Note that this tries to emulate the scaled matmul.
    x16_ans = np.matmul(x32_lhs.astype(np.float16), x32_rhs.astype(np.float16)) * 2 ** (
        log2_scale
    )

    # Create the float8 data on host before moving it to device.
    # Note here we do not scale during the cast, as the only scaling
    # is preformed during the matmul.
    x8_host_lhs = host_pow2scale_cast_to_fp8(x32_lhs, format_, 0, False)
    x8_host_rhs = host_pow2scale_cast_to_fp8(x32_rhs, format_, 0, False)

    with main_graph:
        instream0 = popxl.h2d_stream(shape, format_, "instream0")
        instream1 = popxl.h2d_stream(shape, format_, "instream1")

        log2_scale_t = popxl.variable(log2_scale, dtype=popxl.int32, name="log2_scale")

        x8_lhs = ops.host_load(instream0, "x8_lhs")
        x8_rhs = ops.host_load(instream1, "x8_rhs")

        assert x8_lhs.dtype == format_
        assert x8_rhs.dtype == format_

        x16 = ops.matmul_pow2scaled(x8_lhs, x8_rhs, log2_scale_t)

        # The output of a float8 matmul is always float16
        assert x16.dtype == popxl.float16

        outstream = popxl.d2h_stream(x16.shape, dtype=popxl.float16, name="out_stream")
        ops.host_store(outstream, x16)

    with popxl.Session(ir, "ipu_model") as session:
        outputs = session.run({instream0: x8_host_lhs, instream1: x8_host_rhs})

    result = outputs[outstream]
    assert result.dtype == np.float16

    # Conversion from FP16 to FP8 is lossy so we
    # tolerate some inaccuracy
    np.testing.assert_almost_equal(result, x16_ans, decimal=1)


@pytest.mark.parametrize(
    "format_", [popart.DataType.FLOAT8_143, popart.DataType.FLOAT8_152]
)
def test_matmul_pow2_scaled_raises(format_):
    """Test that the matmul_pow2scaled op raises for some invalid
    combinations on input, such as bad types"""

    ir = popxl.Ir()
    main_graph = ir.main_graph
    shape = (2, 2)

    x16 = np.random.rand(*shape).astype(np.float16)

    dtype = (
        popxl.float8_143 if format_ == popart.DataType.FLOAT8_143 else popxl.float8_152
    )

    with main_graph:
        instream0 = popxl.h2d_stream(shape, dtype, "instream0")
        instream1 = popxl.h2d_stream(shape, dtype, "instream1")
        x8_lhs = ops.host_load(instream0, "x8_lhs")
        x8_rhs = ops.host_load(instream1, "x8_rhs")

        # raises on non FP8 tensors
        lhs_invalid = popxl.constant(x16, dtype=popxl.float16)
        rhs_invalid = popxl.constant(x16, dtype=popxl.float16)
        with pytest.raises(TypeError):
            ops.matmul_pow2scaled(
                lhs_invalid, rhs_invalid, popxl.constant(0, dtype=popxl.int32)
            )

        # raises where the log2 scale is non-scalar
        with pytest.raises(ValueError):
            ops.matmul_pow2scaled(
                x8_lhs,
                x8_rhs,
                rhs_invalid,
                popxl.constant([1, 2, 3], dtype=popxl.int32),
            )

        # raises where log2 scale is not int32
        with pytest.raises(TypeError):
            ops.matmul_pow2scaled(
                x8_lhs, x8_rhs, popxl.variable(3.14, dtype=popxl.float32)
            )


@pytest.mark.parametrize("log2_scale", [-50, 100])
def test_raise_on_log2scale_not_in_range(log2_scale):
    """Test that a poplar runtime error is raised if the log2scale tensor
    contains a value outside of a 6 bit signed integer range."""
    shape = (2, 2)
    format_ = popxl.float8_143

    ir = popxl.Ir()

    # Explicitly set set the option to throw if the log2scale tensor is not in range
    opts = ir._pb_ir.getSessionOptions()
    opts.throwIfLog2ScaleTensorNotInRange = True

    main_graph = ir.main_graph
    x32_lhs: np.ndarray = np.random.rand(*shape)
    x32_rhs: np.ndarray = np.random.rand(*shape)

    x8_host_lhs = host_pow2scale_cast_to_fp8(x32_lhs, format_, 0, False)
    x8_host_rhs = host_pow2scale_cast_to_fp8(x32_rhs, format_, 0, False)

    with main_graph:
        instream0 = popxl.h2d_stream(shape, format_, "instream0")
        instream1 = popxl.h2d_stream(shape, format_, "instream1")

        log2_scale_t = popxl.variable(log2_scale, dtype=popxl.int32, name="log2_scale")

        x8_lhs = ops.host_load(instream0, "x8_lhs")
        x8_rhs = ops.host_load(instream1, "x8_lhs")

        assert x8_lhs.dtype == format_
        assert x8_rhs.dtype == format_

        result = ops.matmul_pow2scaled(x8_lhs, x8_rhs, log2_scale_t)

        # print so we don't prune
        ops.print_tensor(result)

    with popxl.Session(ir, "ipu_model") as session:
        with pytest.raises(popart.poplar_application_runtime_error):
            session.run({instream0: x8_host_lhs, instream1: x8_host_rhs})

    # If we set the option to throw to false, this should run fine
    opts.throwIfLog2ScaleTensorNotInRange = False
    with popxl.Session(ir, "ipu_model") as session:
        session.run({instream0: x8_host_lhs, instream1: x8_host_rhs})
