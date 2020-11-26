# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# Test 1D, 2D and 3D multiconvs, and all permutations thereof
dim_values = [1, 2, 3]


@pytest.mark.parametrize("dims", itertools.product(dim_values, dim_values))
def test_multiconv_infer(op_tester, dims):
    # Shape info for conv0
    bs0 = 1
    chans_in0 = 6
    chans_out0 = 9
    size0 = 4
    kernel_size0 = 3
    dil0 = [1]
    pad0 = [2]  # upper and lower
    stride0 = [1]

    # Shape info for conv1
    bs1 = 2
    chans_in1 = 3
    chans_out1 = 4
    size1 = 5
    kernel_size1 = 2
    dil1 = [2]
    pad1 = [0]  # upper and lower
    stride1 = [1]

    data0 = np.random.rand(bs0, chans_in0,
                           *([size0] * dims[0])).astype(np.float32)
    data1 = np.random.rand(bs1, chans_in1,
                           *([size1] * dims[1])).astype(np.float32)

    filt0 = np.random.rand(chans_out0, chans_in0,
                           *([kernel_size0] * dims[0])).astype(np.float32)
    filt1 = np.random.rand(chans_out1, chans_in1,
                           *([kernel_size1] * dims[1])).astype(np.float32)
    bias1 = np.random.rand(chans_out1).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(data0)
        d1 = builder.addInputTensor(data1)
        f0 = builder.addInputTensor(filt0)
        f1 = builder.addInitializedInputTensor(filt1)
        b1 = builder.addInitializedInputTensor(bias1)
        [c0, c1] = builder.aiGraphcore.multiconv(
            [[d0, f0], [d1, f1, b1]],
            dilations=[dil0 * dims[0], dil1 * dims[1]],
            pads=[pad0 * 2 * dims[0], pad1 * 2 * dims[1]],
            strides=[stride0 * dims[0], stride1 * dims[1]])
        return [c0, c1]

    def reference(ref_data):
        d0 = torch.tensor(data0)
        d1 = torch.tensor(data1)
        conv0 = _torch_convolution(dims[0],
                                   [chans_in0, chans_out0, kernel_size0],
                                   dilation=dil0,
                                   padding=pad0,
                                   stride=stride0)
        conv1 = _torch_convolution(dims[1],
                                   [chans_in1, chans_out1, kernel_size1],
                                   dilation=dil1,
                                   padding=pad1,
                                   stride=stride1)
        conv0.weight.data = torch.tensor(filt0)
        conv0.bias.data = torch.tensor([0.0 for i in range(chans_out0)])
        conv1.weight.data = torch.tensor(filt1)
        conv1.bias.data = torch.tensor(bias1)
        c0 = conv0(d0)
        c1 = conv1(d1)
        return [c0, c1]

    op_tester.run(init_builder, reference, step_type='infer')


def test_multiconv_infer_default(op_tester):
    bs0 = 1
    chans_in0 = 6
    chans_out0 = 9
    size0 = 4
    kernel_size0 = 3

    bs1 = 2
    chans_in1 = 3
    chans_out1 = 4
    size1 = 5
    kernel_size1 = 2

    data0 = np.random.rand(bs0, chans_in0, size0, size0).astype(np.float32)
    data1 = np.random.rand(bs1, chans_in1, size1, size1).astype(np.float32)

    filt0 = np.random.rand(chans_out0, chans_in0, kernel_size0,
                           kernel_size0).astype(np.float32)
    filt1 = np.random.rand(chans_out1, chans_in1, kernel_size1,
                           kernel_size1).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(data0)
        d1 = builder.addInputTensor(data1)
        f0 = builder.addInputTensor(filt0)
        f1 = builder.addInitializedInputTensor(filt1)
        [c0, c1] = builder.aiGraphcore.multiconv([[d0, f0], [d1, f1]])
        return [c0, c1]

    def reference(ref_data):
        d0 = torch.tensor(data0)
        d1 = torch.tensor(data1)
        conv0 = torch.nn.Conv2d(chans_in0, chans_out0, kernel_size0)
        conv1 = torch.nn.Conv2d(chans_in1, chans_out1, kernel_size1)
        conv0.weight.data = torch.tensor(filt0)
        conv0.bias.data = torch.tensor([0.0 for i in range(chans_out0)])
        conv1.weight.data = torch.tensor(filt1)
        conv1.bias.data = torch.tensor([0.0 for i in range(chans_out1)])
        c0 = conv0(d0)
        c1 = conv1(d1)
        return [c0, c1]

    op_tester.run(init_builder, reference, step_type='infer')


@pytest.mark.parametrize("dims", itertools.product(dim_values, dim_values))
def test_multiconv_train_default_parameters_and_conv_options(op_tester, dims):
    """
    Add a multiconv op, comprised of two convolutions, and test for training.
    Strides, Pads, Dilations, and all convolution options are the defaults
    """
    bs0 = 1
    chans_in0 = 6
    chans_out0 = 9
    size0 = 6
    kernel_size0 = 3

    bs1 = 2
    chans_in1 = 3
    chans_out1 = 4
    size1 = 5
    kernel_size1 = 2

    data0 = np.random.rand(bs0, chans_in0,
                           *([size0] * dims[0])).astype(np.float32)
    data1 = np.random.rand(bs1, chans_in1,
                           *([size1] * dims[1])).astype(np.float32)

    filt0 = np.random.rand(chans_out0, chans_in0,
                           *([kernel_size0] * dims[0])).astype(np.float32)
    filt1 = np.random.rand(chans_out1, chans_in1,
                           *([kernel_size1] * dims[1])).astype(np.float32)
    bias1 = np.random.rand(chans_out1).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(data0)
        d1 = builder.addInputTensor(data1)
        f0 = builder.addInputTensor(filt0)
        f1 = builder.addInitializedInputTensor(filt1)
        b1 = builder.addInitializedInputTensor(bias1)
        [c0, c1] = builder.aiGraphcore.multiconv([[d0, f0], [d1, f1, b1]])
        sumc0 = builder.aiOnnx.reducesum([c0], keepdims=0)
        sumc1 = builder.aiOnnx.reducesum([c1], keepdims=0)
        out = builder.aiOnnx.add([sumc0, sumc1])
        return [
            out,
            popart.reservedGradientPrefix() + d0,
            popart.reservedGradientPrefix() + d1,
            popart.reservedGradientPrefix() + f0,
            popart.reservedGradientPrefix() + f1,
            popart.reservedGradientPrefix() + b1
        ]

    def reference(ref_data):
        d0 = torch.tensor(data0, requires_grad=True)
        d1 = torch.tensor(data1, requires_grad=True)
        conv0 = _torch_convolution(
            dims[0], [chans_in0, chans_out0, kernel_size0], None, None, None)
        conv1 = _torch_convolution(
            dims[1], [chans_in1, chans_out1, kernel_size1], None, None, None)
        conv0.weight.data = torch.tensor(filt0)
        conv0.bias.data = torch.tensor([0.0 for i in range(chans_out0)])
        conv1.weight.data = torch.tensor(filt1)
        conv1.bias.data = torch.tensor(bias1)
        c0 = conv0(d0)
        c1 = conv1(d1)
        out = torch.sum(c0) + torch.sum(c1)
        out.backward()
        return [
            out, d0.grad, d1.grad, conv0.weight.grad, conv1.weight.grad,
            conv1.bias.grad
        ]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


def test_multiconv_train_all_conv_options(op_tester):
    """
    Add a multiconv op, comprised of two convolutions, and test for training.
    Strides, Pads, Dilations, are the defaults. Pass in non-default values
    for all convolution options
    """
    bs0 = 1
    chans_in0 = 6
    chans_out0 = 9
    size0 = 6
    kernel_size0 = 3

    bs1 = 2
    chans_in1 = 3
    chans_out1 = 4
    size1 = 5
    kernel_size1 = 2

    data0 = np.random.rand(bs0, chans_in0, size0, size0).astype(np.float32)
    data1 = np.random.rand(bs1, chans_in1, size1, size1).astype(np.float32)

    filt0 = np.random.rand(chans_out0, chans_in0, kernel_size0,
                           kernel_size0).astype(np.float32)
    filt1 = np.random.rand(chans_out1, chans_in1, kernel_size1,
                           kernel_size1).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(data0)
        d1 = builder.addInputTensor(data1)
        f0 = builder.addInputTensor(filt0)
        f1 = builder.addInitializedInputTensor(filt1)
        [c0, c1] = builder.aiGraphcore.multiconv(
            [[d0, f0], [d1, f1]],
            planType="serial",
            cycleBackOff=0.3,
            perConvReservedTiles=100,
            partialsTypes=["float", "float"],
            availableMemoryProportions=[0.9, 0.2])
        sumc0 = builder.aiOnnx.reducesum([c0], keepdims=0)
        sumc1 = builder.aiOnnx.reducesum([c1], keepdims=0)
        out = builder.aiOnnx.add([sumc0, sumc1])
        return [
            out,
            popart.reservedGradientPrefix() + d0,
            popart.reservedGradientPrefix() + d1,
            popart.reservedGradientPrefix() + f0,
            popart.reservedGradientPrefix() + f1
        ]

    def reference(ref_data):
        d0 = torch.tensor(data0, requires_grad=True)
        d1 = torch.tensor(data1, requires_grad=True)
        conv0 = torch.nn.Conv2d(chans_in0, chans_out0, kernel_size0)
        conv1 = torch.nn.Conv2d(chans_in1, chans_out1, kernel_size1)
        conv0.weight.data = torch.tensor(filt0)
        conv0.bias.data = torch.tensor([0.0 for i in range(chans_out0)])
        conv1.weight.data = torch.tensor(filt1)
        conv1.bias.data = torch.tensor([0.0 for i in range(chans_out1)])
        c0 = conv0(d0)
        c1 = conv1(d1)
        out = torch.sum(c0) + torch.sum(c1)
        out.backward()
        return [out, d0.grad, d1.grad, conv0.weight.grad, conv1.weight.grad]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


def test_multiconv_bad_options():

    builder = popart.Builder()
    d0 = builder.addInputTensor("FLOAT", [1, 2, 4, 4])
    f0 = builder.addInputTensor("FLOAT", [10, 2, 3, 3])

    with pytest.raises(popart.popart_exception) as e_info:
        [c0, c1] = builder.aiGraphcore.multiconv([[d0, f0], [d0, f0]],
                                                 dilations=[[1, 1], [2, 2, 2]])
    assert (e_info.value.args[0] ==
            "Length of dilations vector 3 != number of spatial dimensions 2")

    with pytest.raises(popart.popart_exception) as e_info:
        [c0, c1] = builder.aiGraphcore.multiconv([[d0, f0], [d0, f0]],
                                                 dilations=[[1, 1], [2, 2],
                                                            [3, 3]])
    assert (e_info.value.args[0].endswith(
        "number of dilations parameter sets (3) does not match the number of input sets (2)"
    ))

    with pytest.raises(popart.popart_exception) as e_info:
        [c0, c1] = builder.aiGraphcore.multiconv([[d0], [d0, f0]])
    assert (e_info.value.args[0].endswith(
        "must have at least two inputs - data and weights"))

    with pytest.raises(popart.popart_exception) as e_info:
        [c0, c1] = builder.aiGraphcore.multiconv([[d0, f0, d0, f0], [d0, f0]])
    assert (e_info.value.args[0].endswith(
        "can have at most three inputs - data, weights, and bias"))


def test_multiconv_shape_inference():
    builder = popart.Builder()
    d0 = builder.addInputTensor("FLOAT", [1, 6, 4])
    f0 = builder.addInputTensor("FLOAT", [9, 6, 3])
    d1 = builder.addInputTensor("FLOAT", [2, 3, 5, 5])
    f1 = builder.addInputTensor("FLOAT", [4, 3, 2, 2])

    [c0, c1] = builder.aiGraphcore.multiconv([[d0, f0], [d1, f1]])
    c0_expected_shape = [1, 9, 2]
    c1_expected_shape = [2, 4, 4, 4]
    assert builder.getTensorShape(c0) == c0_expected_shape
    assert builder.getTensorShape(c1) == c1_expected_shape

    [c2] = builder.aiGraphcore.multiconv([[d1, f1]],
                                         pads=[[0, 0, 0, 0]],
                                         dilations=[[2, 2]],
                                         strides=[[1, 1]])
    c2_expected_shape = [2, 4, 3, 3]
    assert builder.getTensorShape(c2) == c2_expected_shape


def _torch_convolution(numDims, shapes, dilation, padding, stride):
    if numDims == 1:
        if dilation == None and padding == None and stride == None:
            conv = torch.nn.Conv1d(*shapes)
        else:
            conv = torch.nn.Conv1d(*shapes,
                                   dilation=dilation * numDims,
                                   padding=padding * numDims,
                                   stride=stride * numDims)
    elif numDims == 2:
        if dilation == None and padding == None and stride == None:
            conv = torch.nn.Conv2d(*shapes)
        else:
            conv = torch.nn.Conv2d(*shapes,
                                   dilation=dilation * numDims,
                                   padding=padding * numDims,
                                   stride=stride * numDims)
    elif numDims == 3:
        if dilation == None and padding == None and stride == None:
            conv = torch.nn.Conv3d(*shapes)
        else:
            conv = torch.nn.Conv3d(*shapes,
                                   dilation=dilation * numDims,
                                   padding=padding * numDims,
                                   stride=stride * numDims)
    else:
        raise Exception("Torch convoltion dims can be only 1, 2 or 3")

    return conv
