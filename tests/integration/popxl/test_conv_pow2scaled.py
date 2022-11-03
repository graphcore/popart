# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest

import popxl.ops as ops
import popxl

import numpy as np
import torch
import torch.nn as nn
import popart

from popxl.utils import host_cast_then_pow2scale, host_pow2scale_then_cast


@pytest.mark.parametrize("format_", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("log2_scale", [-1, 0, 1])
def test_fp8_conv(format_, log2_scale):
    """Test that the conv_pow2scaled op returns approximately similar results to a
    pytorch conv that is then scaled by pow2(log2_scale) for both float8 formats.
    """
    # arbitrary conv parameters
    batch_size = 2
    in_channel = 10
    out_channel = 20
    height = 50
    width = 50
    h_kernel = 3
    w_kernel = 3
    strides = (1, 1)
    group = 1

    def cast_to_float8_and_back(x, fmt):
        """Cast a numpy array to float8, and then back to float16 to
        reduce rounding error when performing fused ops later"""
        temp = host_pow2scale_then_cast(x, fmt, 0, False)
        return host_cast_then_pow2scale(temp, popxl.float32, 0).astype(np.float16)

    t = np.random.rand(batch_size, in_channel, height, width)
    weight = np.random.rand(out_channel, in_channel // group, h_kernel, w_kernel)

    # cast to FP8 and back again
    t = cast_to_float8_and_back(t, format_)
    weight = cast_to_float8_and_back(weight, format_)

    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        log2_scale_t = popxl.variable(log2_scale, dtype=popxl.int32)
        # host load
        input0 = popxl.h2d_stream(
            [batch_size, in_channel, height, width],
            popxl.float16,
            name="in_stream_0",
        )
        a = ops.host_load(input0, "a")
        input1 = popxl.h2d_stream(
            [out_channel, int(in_channel / group), h_kernel, w_kernel],
            popxl.float16,
            name="in_stream_1",
        )
        b = ops.host_load(input1, "b")

        # Cast to fp8 on device before conv
        a_fp8 = ops.pow2scale_then_cast(
            a, data_type=format_, log2_scale=popxl.constant(0)
        )
        b_fp8 = ops.pow2scale_then_cast(
            b, data_type=format_, log2_scale=popxl.constant(0)
        )

        assert a_fp8.dtype == format_

        # FP8s and FP16s cannot be mixed on matmuls so
        # both t and weight must be FP8s in the conv
        o = ops.conv_pow2scaled(a_fp8, b_fp8, log2_scale_t, stride=strides)

        assert o.dtype == popxl.float16

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
        ops.host_store(o_d2h, o)

    with popxl.Session(ir, "ipu_model") as session:
        outputs = session.run({input0: t, input1: weight})

    # Do a conv in torch with the same parameters to compare for correctness.
    torch_t = torch.tensor(t).type(torch.float32)
    torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
    torch_outputs = nn.Conv2d(
        in_channel,
        out_channel,
        (h_kernel, w_kernel),
        tuple(strides),
        groups=group,
        bias=False,
    )
    torch_outputs.weight = torch_weight
    torch_output_data = torch_outputs(torch_t)
    # compare the result between PopXL and torch. Note that the torch
    # result is scaled by 2**log2_scale to mimic the behaviour of conv_pow2scaled.
    torch_result = torch_output_data.detach().numpy() * (2 ** log2_scale)

    # Conversion from FP16 to FP8 is lossy so we tolerate some inaccuracy
    np.testing.assert_almost_equal(outputs[o_d2h], torch_result, decimal=1)


@pytest.mark.parametrize("format_", [popxl.float8_143, popxl.float8_152])
def test_raises_on_invalid_inputs(format_):
    """Test that the conv_pow2scaled op raises exceptions when
    the inputs are invalid for a float8 convolution"""
    ir = popxl.Ir()
    main_graph = ir.main_graph
    shape = (1, 1, 10, 10)

    t = np.random.rand(*shape)
    weight = np.random.rand(*shape)

    with main_graph:
        input0 = popxl.h2d_stream(
            list(shape),
            format_,
            name="in_stream_0",
        )
        input1 = popxl.h2d_stream(
            list(shape),
            format_,
            name="in_stream_1",
        )
        a = ops.host_load(input0, "a")
        b = ops.host_load(input1, "b")

        # raises on non-FP8 tensors
        inputs_invalid = popxl.constant(t, dtype=popxl.float32)
        weights_invalid = popxl.variable(weight, dtype=popxl.float32)
        log2_scale = popxl.constant(0, dtype=popxl.int32)
        with pytest.raises(TypeError):
            ops.conv_pow2scaled(inputs_invalid, weights_invalid, log2_scale)

        # raises where the log2 scale is non-scalar
        with pytest.raises(ValueError, match="must be a scalar tensor"):
            ops.conv_pow2scaled(
                a,
                b,
                popxl.constant([1, 2, 3], dtype=popxl.int32),
            )

        # raises where log2 scale is not int32
        log2_scale_int8 = popxl.constant(1, dtype=popxl.int8)
        with pytest.raises(TypeError, match="must be of type popxl.int32"):
            ops.conv_pow2scaled(a, b, log2_scale_int8)


@pytest.mark.parametrize("log2_scale", [-50, 100])
def test_raise_on_log2scale_not_in_range(log2_scale):
    """Test that a poplar runtime error is raised if the log2scale tensor
    contains a value outside of a 6 bit signed integer range."""
    # arbitrary conv parameters
    batch_size = 2
    in_channel = 10
    out_channel = 20
    height = 50
    width = 50
    h_kernel = 3
    w_kernel = 3
    group = 1

    format_ = popxl.float8_143

    ir = popxl.Ir()

    # Explicitly set set the option to throw if the log2scale tensor is not in range
    opts = ir._pb_ir.getSessionOptions()
    opts.throwIfLog2ScaleTensorNotInRange = True

    t = np.random.rand(batch_size, in_channel, height, width)
    weight = np.random.rand(out_channel, in_channel // group, h_kernel, w_kernel)
    main_graph = ir.main_graph
    with main_graph:
        input_t = popxl.variable(t, dtype=format_, log2_scale=0, name="input")
        weight_t = popxl.variable(weight, dtype=format_, log2_scale=0, name="weights")
        log2_scale_t = popxl.constant(log2_scale, dtype=popxl.int32, name="log2scale")

        o = ops.conv_pow2scaled(input_t, weight_t, log2_scale_t)

        # print the tensor to prevent AliasZeroCopy from disabling the conv op
        ops.print_tensor(o)

    with popxl.Session(ir, "ipu_model") as session:
        with pytest.raises(popart.poplar_application_runtime_error):
            session.run()

    # If we set the option to throw to false, this should run fine
    opts.throwIfLog2ScaleTensorNotInRange = False
    with popxl.Session(ir, "ipu_model") as session:
        session.run()
