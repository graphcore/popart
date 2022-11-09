# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import pytest
import popart._internal.ir as _ir
from utils import contains_op_of_type

batch_size = 1
in_channel = 4
out_channel = 4
height = 1
width = 1
h_kernel = 1
w_kernel = 1
group = 1


@pytest.mark.parametrize("pad_type", ("not_set", "same_upper", "same_lower", "valid"))
def test_conv_transpose(pad_type):
    """Test that we can successfully add the conv transpose op to the IR"""

    ir = popxl.Ir()
    g = ir.main_graph
    with g:
        t = popxl.variable(np.random.rand(batch_size, in_channel, height, width))
        weight = popxl.variable(
            np.random.rand(out_channel, int(in_channel / group), h_kernel, w_kernel)
        )
        _ = ops.conv_transpose(t, weight, groups=group, pad_type=pad_type)
    assert len(g.tensors) == 3
    assert len(g.variables) == 2
    assert contains_op_of_type("ConvTranspose", _ir.op.ConvTransposeOp, g)


def test_output_shape():
    """Conv transpose will either work out the output shape for you, or it can be set manually.
    For both case, check this is set correctly on the output tensor.
    """
    output_shape = (batch_size, out_channel, 2, 2)

    strides = (1, 1)
    dilations = (1, 1)
    # add a pixel at both ends of each spatial dimension
    pads = (1, 1, 1, 1)
    # Values must be less than the correspodning stride/dilation dimension
    output_padding = (0, 0, 0, 0)

    ir = popxl.Ir()
    with ir.main_graph:
        t = popxl.variable(np.random.rand(batch_size, in_channel, height, width))
        weight = popxl.variable(
            np.random.rand(out_channel, int(in_channel / group), h_kernel, w_kernel)
        )
        # We explicitly set the output shape, so check its been set correctly on the output tensor
        out_explicit = ops.conv_transpose(
            t, weight, groups=group, output_shape=output_shape
        )

        assert out_explicit.shape == output_shape

        # Now, don't set the output shape, but pass the pads parameter should work
        # out the output shape automatically
        out_auto = ops.conv_transpose(
            t,
            weight,
            groups=group,
            stride=strides,
            dilation=dilations,
            padding=pads,
            pad_type="not_set",
        )

        input_shape = (height, width)
        kernel_shape = (h_kernel, w_kernel)

        # Compute the expected shape according to
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
        expected_shape = [batch_size, out_channel]
        for i in range(2):
            x = (
                strides[i] * (input_shape[i] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - pads[0]
                - pads[1]
            )
            expected_shape.append(x)

        assert out_auto.shape == tuple(expected_shape)
