# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import pytest
from popxl import float8_143, float8_152
from popxl.utils import host_cast_then_pow2scale, host_pow2scale_then_cast
from utils import get_float8_data, get_representable_float_8_np_array

import popart


@pytest.mark.parametrize(
    "float8_format",
    [float8_143, float8_152],
)
@pytest.mark.parametrize(
    "log2_scale",
    [-4, -1, 0, 1, 4],
)
def test_pow2scale_then_cast(float8_format: popxl.dtype, log2_scale: int):
    """Test casting float16 to float8 on device vs host.

    Args:
        float8_format (popxl.dtype): Format 143 or 152
        log2_scale (int): log scaling to use
    """

    shape = [10, 10, 10]

    # Note this array may contain the equivalent nan value for the given format,
    # so we must set nan_on_overflow = True on host.
    d1 = get_representable_float_8_np_array(shape, float8_format, log2_scale)

    d2 = np.array(log2_scale, np.int32)
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        input0 = popxl.h2d_stream(
            d1.shape,
            popxl.float16,
            name="in_stream_0",
        )
        input1 = popxl.h2d_stream(
            d2.shape,
            popxl.int32,
            name="in_stream_0",
        )
        input_ = ops.host_load(input0, "input_")
        log2_scale_tensor = ops.host_load(input1, "log2_scale_tensor")

        cast = ops.pow2scale_then_cast(input_, log2_scale_tensor, float8_format)

        o_d2h = popxl.d2h_stream(cast.shape, cast.dtype, name="out_stream")
        ops.host_store(o_d2h, cast)
        with popxl.Session(ir, "ipu_model") as session:
            for i in range(5):
                print(f"Step {i}")

                outputs = session.run({input0: d1.astype(np.float16), input1: d2})

                # These should match exactly as the host conversion should be the same as
                # the device conversion.
                # 1. uint32 tensor returned from device, converted to popxl float8 numpy type
                array_1 = outputs[o_d2h]
                # 2. Original fp32 data converted on the host as popxl float8 numpy type
                array_2 = host_pow2scale_then_cast(
                    d1.astype(np.float32),
                    float8_format,
                    log2_scale,
                    True,
                )
                np.testing.assert_equal(array_1, array_2)

                # regenerate array
                d1 = get_representable_float_8_np_array(
                    shape, float8_format, log2_scale
                )


@pytest.mark.parametrize(
    "float8_format",
    [float8_143, float8_152],
)
@pytest.mark.parametrize(
    "log2_scale",
    [-4, -1, 0, 1, 4],
)
def test_cast_then_pow2scale(float8_format: popxl.dtype, log2_scale: int):
    """Test casting float8 to float16 on device vs host.

    Args:
        float8_format (popxl.dtype): Format 143 or 152
        log2_scale (int): log scaling to use
    """

    shape = [10, 10, 10]
    # Convert to float8 data on host. This is the baseline data we use for
    # both host and device conversion.

    # Note this array may contain the equivalent nan value for the given format,
    # so we must set nan_on_overflow = True on host.
    d1_float8 = get_float8_data(float8_format, log2_scale, shape)

    # When converting back again we will negate the scale.
    d2 = np.array(-log2_scale, np.int32)
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        input0 = popxl.h2d_stream(
            d1_float8.shape,
            float8_format,
            name="in_stream_0",
        )
        input1 = popxl.h2d_stream(
            d2.shape,
            popxl.int32,
            name="in_stream_0",
        )
        input_ = ops.host_load(input0, "input_")
        log2_scale_tensor = ops.host_load(input1, "log2_scale_tensor")

        cast = ops.cast_then_pow2scale(input_, log2_scale_tensor, popxl.float16)

        o_d2h = popxl.d2h_stream(cast.shape, cast.dtype, name="out_stream")
        ops.host_store(o_d2h, cast)
        with popxl.Session(ir, "ipu_model") as session:
            for i in range(5):
                print(f"Step {i}")
                outputs = session.run({input0: d1_float8, input1: d2})

                # These should match exactly as the host conversion should be the same as
                # the device conversion.
                # 1. Float16 data returned to the host from device
                array_1 = outputs[o_d2h].astype(np.float32)
                # 2. The float8 array converted back to float32 on host.
                array_2 = host_cast_then_pow2scale(
                    d1_float8, popxl.float32, -log2_scale
                )
                np.testing.assert_equal(array_1, array_2)

                # regenerate array
                d1_float8 = get_float8_data(float8_format, log2_scale, shape)


# (cast op, dtype casted from, dtype casted to)
@pytest.mark.parametrize(
    "cast_op_and_types",
    [
        (ops.cast_then_pow2scale, float8_143, popxl.float16),
        (ops.pow2scale_then_cast, popxl.float16, float8_143),
    ],
)
@pytest.mark.parametrize("log2_scale", [-50, 100])
def test_raise_on_log2scale_not_in_range(cast_op_and_types, log2_scale):
    """Test that a poplar runtime error is raised if the log2scale tensor
    contains a value outside of a 6 bit signed integer range when casting."""
    ir = popxl.Ir()
    cast_op, from_dtype, to_dtype = cast_op_and_types

    # Explicitly set set the option to throw if the log2scale tensor is not in range
    opts = ir._pb_ir.getSessionOptions()
    opts.throwIfLog2ScaleTensorNotInRange = True

    t = np.random.rand(10, 10)
    with ir.main_graph:
        input_t = popxl.variable(t, dtype=from_dtype, name="input")
        log2_scale_t = popxl.constant(log2_scale, dtype=popxl.int32, name="log2scale")

        o = cast_op(input_t, log2_scale_t, data_type=to_dtype)

        # print the tensor to prevent AliasZeroCopy from disabling the cast op
        ops.print_tensor(o)

    with popxl.Session(ir, "ipu_model") as session:
        with pytest.raises(popart.poplar_application_runtime_error):
            session.run()

    # If we set the option to throw to false, this should run fine
    opts.throwIfLog2ScaleTensorNotInRange = False
    with popxl.Session(ir, "ipu_model") as session:
        session.run()
