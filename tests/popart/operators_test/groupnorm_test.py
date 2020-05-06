# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import os
from op_tester import op_tester

# The poplar groups are strided, but the pytorch are not so the
# gradient tensors are different.


def test_groupnorm_0(op_tester):

    epsilon = 1e-05
    num_channels = 4
    num_groups = 2

    # create test data
    d1 = 0.1 * np.random.randn(2, 4, 2, 2).astype(np.float32) - 0.02

    # Init with same values as pytorch
    b = np.zeros(num_channels).astype(np.float32)
    scale = np.ones(num_channels).astype(np.float32)

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_invstd) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups, epsilon)
        builder.addOutputTensor(o_y)
        return [
            o_y, o_mean, o_invstd,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + iB,
            popart.reservedGradientPrefix() + iScale,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):

        # Switch the rows to simulate the striding
        d1[0][[2, 1]] = d1[0][[1, 2]]
        d1[1][[2, 1]] = d1[1][[1, 2]]

        _input = torch.tensor(d1, requires_grad=True)

        m = torch.nn.GroupNorm(num_groups, num_channels, eps=epsilon)

        m.train()
        _y = m(_input)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        t = _y[0][2].clone()
        _y[0][2] = _y[0][1]
        _y[0][1] = t

        t = _y[1][2].clone()
        _y[1][2] = _y[1][1]
        _y[1][1] = t

        # Get a view of input by channel
        v = _input.view(4, -1)
        # Get the mean and inv std dev
        _mean = v.mean(-1)
        _invstd = 1 / torch.sqrt(v.var(-1, unbiased=False) + epsilon)

        return [_y, _mean, _invstd, None, None, None, d__o]

    op_tester.patterns = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_groupnorm_1(op_tester):
    epsilon = 1e-05
    num_channels = 4
    num_groups = 1

    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]],
                    [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0.5, 1]], [[1, 0], [0, 1]],
                    [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    # Init with same values as pytorch
    b = np.zeros(num_channels).astype(np.float32)
    scale = np.ones(num_channels).astype(np.float32)

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_invstd) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups)
        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_invstd)
        return [
            o_y, o_mean, o_invstd,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + iScale,
            popart.reservedGradientPrefix() + iB,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):

        _input = torch.tensor(d1, requires_grad=True)

        m = torch.nn.GroupNorm(num_groups, num_channels)

        m.train()
        _y = m(_input)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        # Get a view of input by channel
        v = _input.view(2, -1)
        # Get the mean and inv std dev
        _mean = v.mean(-1)
        _invstd = 1 / torch.sqrt(v.var(-1, unbiased=False) + epsilon)

        return [
            _y, _mean, _invstd, _input.grad, m.weight.grad, m.bias.grad, d__o
        ]

    op_tester.patterns = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_groupnorm_2(op_tester):
    epsilon = 1e-05
    num_channels = 6
    num_groups = 6

    # create test data
    d1 = np.random.rand(5, num_channels, 2, 2).astype(np.float32) * 10
    b = np.random.rand(num_channels).astype(np.float32)
    scale = np.random.rand(num_channels).astype(np.float32)

    # Relax the relative tolerance as small numbers lose precision
    op_tester.rtol = 1e-04
    op_tester.atol = 1e-06

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_var) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups)
        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_var)
        return [
            o_y, o_mean, o_var,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + iScale,
            popart.reservedGradientPrefix() + iB,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):

        _input = torch.tensor(d1, requires_grad=True)
        _weight = torch.tensor(scale, requires_grad=True)
        _bias = torch.tensor(b, requires_grad=True)

        m = torch.nn.GroupNorm(num_groups, num_channels)
        m.weight.data.copy_(_weight)
        m.bias.data.copy_(_bias)

        m.train()
        _y = m(_input)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        # Get a view of input by channel
        v = _input.view(30, -1)
        # Get the mean and var
        _mean = v.mean(-1)

        # Get the mean and inv std dev
        _invstd = 1 / torch.sqrt(v.var(-1, unbiased=False) + epsilon)

        return [
            _y, _mean, _invstd, _input.grad, m.weight.grad, m.bias.grad, d__o
        ]

    op_tester.patterns = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


# test with large epsilon
def test_groupnorm_3(op_tester):
    epsilon = 0.2
    num_channels = 4
    num_groups = 1

    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]],
                    [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0.5, 1]], [[1, 0], [0, 1]],
                    [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    b = np.zeros(num_channels).astype(np.float32)
    scale = np.ones(num_channels).astype(np.float32)

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_invstd) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups, epsilon)
        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_invstd)
        return [
            o_y, o_mean, o_invstd,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + iScale,
            popart.reservedGradientPrefix() + iB,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):

        _input = torch.tensor(d1, requires_grad=True)

        m = torch.nn.GroupNorm(num_groups, num_channels, eps=epsilon)

        m.train()
        _y = m(_input)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        # Get a view of input by channel
        v = _input.view(2, -1)
        # Get the mean and var
        _mean = v.mean(-1)

        # Get the mean and inv std dev
        _invstd = 1 / torch.sqrt(v.var(-1, unbiased=False) + epsilon)

        return [
            _y, _mean, _invstd, _input.grad, m.weight.grad, m.bias.grad, d__o
        ]

    op_tester.patterns = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


# Test vs. numpy as well, just in case and at fp16
def test_groupnorm_4(op_tester):
    epsilon = 1e-03
    num_channels = 4
    num_groups = 2
    # create test data
    d1 = 0.1 * np.random.rand(2, num_channels, 2, 2).astype(np.float16) - 0.02

    b = np.zeros(num_channels).astype(np.float16)
    scale = np.ones(num_channels).astype(np.float16)

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_invstd) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups, epsilon)
        builder.addOutputTensor(o_y)
        return [
            o_y, o_mean, o_invstd,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + iB,
            popart.reservedGradientPrefix() + iScale,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):

        _y, _mean, _invstd = refGroupNormFwd(d1,
                                             scale,
                                             b,
                                             num_groups,
                                             epsilon=epsilon,
                                             data_format="NCHW")

        return [
            _y.astype(np.float16),
            _mean.astype(np.float16),
            _invstd.astype(np.float16), None, None, None, None
        ]

    # We have to relax the tolerances slightly to account for the fact the
    # fp16 outputs might be slightly off.
    op_tester.atol = 1e-6
    op_tester.rtol = 1e-3
    op_tester.patterns = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def refGroupNormFwd(inputs,
                    gamma,
                    beta,
                    groups,
                    mean=None,
                    inv_std_dev=None,
                    epsilon=0.0015,
                    data_format="NHWC"):
    if data_format == "NHWC":
        feature_index = 3
    elif data_format == "NCHW":
        feature_index = 1
    else:
        raise Exception("Unsupported data format " + data_format)

    num_channels = inputs.shape[feature_index]
    group_size = num_channels // groups
    original_shape = inputs.shape

    # Implementation detail - in Poplibs group norm, the groups are not
    # contiguous, but strided - we replicate that here
    # Move the channels to the first dimension for inputs, gamma and beta
    inputs = np.swapaxes(inputs, 0, feature_index)

    reshuffled_inputs = np.empty(inputs.shape, inputs.dtype)
    reshuffled_gamma = np.empty(gamma.shape, gamma.dtype)
    reshuffled_beta = np.empty(beta.shape, beta.dtype)

    for from_idx in range(num_channels):
        to_idx = (from_idx % groups) * group_size + from_idx // groups
        reshuffled_inputs[to_idx] = inputs[from_idx]
        reshuffled_gamma[to_idx] = gamma[from_idx]
        reshuffled_beta[to_idx] = beta[from_idx]
    inputs = np.swapaxes(reshuffled_inputs, 0, feature_index)
    gamma = reshuffled_gamma
    beta = reshuffled_beta

    if feature_index == 1:
        N, C, H, W = inputs.shape
        inputs = np.reshape(inputs, [N, groups, C // groups, H, W])
        gamma = np.reshape(gamma, [1, C, 1, 1])
        beta = np.reshape(beta, [1, C, 1, 1])
        moments_axes = (feature_index + 1, 3, 4)

        if mean is not None and inv_std_dev is not None:
            mean = np.reshape(mean, [N, groups, 1, 1, 1])
            inv_std_dev = np.reshape(inv_std_dev, [N, groups, 1, 1, 1])
    else:
        N, H, W, C = inputs.shape
        inputs = np.reshape(inputs, [N, H, W, groups, C // groups])
        gamma = np.reshape(gamma, [1, 1, 1, C])
        beta = np.reshape(beta, [1, 1, 1, C])
        moments_axes = (1, 2, feature_index + 1)

        if mean is not None and inv_std_dev is not None:
            mean = np.reshape(mean, [N, 1, 1, groups, 1])
            inv_std_dev = np.reshape(inv_std_dev, [N, 1, 1, groups, 1])

    if mean is None and inv_std_dev is None:
        mean = np.mean(inputs, moments_axes, dtype=np.float32, keepdims=True)
        variance = np.mean(np.power(inputs - mean, 2),
                           moments_axes,
                           dtype=np.float32,
                           keepdims=True)
    else:
        variance = np.power(inv_std_dev, -2) - epsilon

    input_whitened = (inputs - mean) * np.power(variance + epsilon, -0.5)
    input_whitened = np.reshape(input_whitened, original_shape)
    output = input_whitened * gamma + beta

    # Undo the shuffle.
    output = np.swapaxes(output, 0, feature_index)

    reshuffled_output = np.empty(output.shape, output.dtype)
    for to_idx in range(num_channels):
        from_idx = (to_idx % groups) * group_size + to_idx // groups
        reshuffled_output[to_idx] = output[from_idx]
    inv_std_dev = np.power(variance + epsilon, -0.5)
    return (np.swapaxes(reshuffled_output, 0, feature_index),
            np.reshape(np.squeeze(mean), (mean.size)),
            np.reshape(np.squeeze(inv_std_dev), (inv_std_dev.size)))


# Test stable group norm algorith - this test fails without it.
def test_groupnorm_5(op_tester):

    epsilon = 1e-05
    num_channels = 4
    num_groups = 2

    np.random.seed(0)
    # create test data
    d1 = np.random.randn(2, 4, 2, 2).astype(np.float32) + 1000

    # Init with same values as pytorch
    b = np.zeros(num_channels).astype(np.float32)
    scale = np.ones(num_channels).astype(np.float32)

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_invstd) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups, epsilon)
        builder.addOutputTensor(o_y)
        return [
            o_y, o_mean, o_invstd,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + iB,
            popart.reservedGradientPrefix() + iScale,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):

        # Switch the rows to simulate the striding
        d1[0][[2, 1]] = d1[0][[1, 2]]
        d1[1][[2, 1]] = d1[1][[1, 2]]

        _input = torch.tensor(d1, requires_grad=True)

        m = torch.nn.GroupNorm(num_groups, num_channels, eps=epsilon)

        m.train()
        _y = m(_input)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        t = _y[0][2].clone()
        _y[0][2] = _y[0][1]
        _y[0][1] = t

        t = _y[1][2].clone()
        _y[1][2] = _y[1][1]
        _y[1][1] = t

        # Get a view of input by channel
        v = _input.view(4, -1)
        # Get the mean and inv std dev
        _mean = v.mean(-1)

        _invstd = 1 / torch.sqrt(v.var(-1, unbiased=False) + epsilon)

        return [_y, _mean, _invstd, None, None, None, d__o]

    # This fails without this flag
    op_tester.options.enableStableNorm = True
    # Calculation is still pretty iffy at such large mean / std dev ratio.
    op_tester.rtol = 1e-3
    op_tester.atol = 1e-5
    op_tester.patterns = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')
