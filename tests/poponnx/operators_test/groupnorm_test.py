import numpy as np
import pytest
import poponnx
import torch
import os
from op_tester import op_tester

# The poplar groups are strided, but the pytorch are not so the
# gradient tensors are different.


def test_groupnorm_0(op_tester):
    # create test data
    d1 = np.random.rand(2, 4, 2, 2).astype(np.float32) * 10

    b = np.zeros(4).astype(np.float32)
    scale = np.ones(4).astype(np.float32)

    epsilon = 1e-05
    num_channels = 4
    num_groups = 2

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_var) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups, epsilon)
        builder.addOutputTensor(o_y)
        return [
            o_y, o_mean, o_var,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + iB,
            poponnx.reservedGradientPrefix() + iScale,
            poponnx.reservedGradientPrefix() + o_y
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
        # Get the mean and var
        _mean = v.mean(-1)
        _var = v.var(-1, unbiased=False)

        return [_y, _mean, _var, None, None, None, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_groupnorm_1(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]],
                    [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0.5, 1]], [[1, 0], [0, 1]],
                    [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    b = np.zeros(4).astype(np.float32)
    scale = np.ones(4).astype(np.float32)

    epsilon = 1e-05
    num_channels = 4
    num_groups = 1

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
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + iScale,
            poponnx.reservedGradientPrefix() + iB,
            poponnx.reservedGradientPrefix() + o_y
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
        # Get the mean and var
        _mean = v.mean(-1)
        _var = v.var(-1, unbiased=False)

        return [_y, _mean, _var, _input.grad, m.weight.grad, m.bias.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_groupnorm_2(op_tester):
    # create test data
    d1 = np.random.rand(5, 6, 2, 2).astype(np.float32) * 10
    b = np.random.rand(6).astype(np.float32)
    scale = np.random.rand(6).astype(np.float32)

    epsilon = 1e-05
    num_channels = 6
    num_groups = 6

    # Relax the relative tolerance as small numbers lose precison
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
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + iScale,
            poponnx.reservedGradientPrefix() + iB,
            poponnx.reservedGradientPrefix() + o_y
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
        _var = v.var(-1, unbiased=False)

        return [_y, _mean, _var, _input.grad, m.weight.grad, m.bias.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


# test will large epsilon
def test_groupnorm_3(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]],
                    [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0.5, 1]], [[1, 0], [0, 1]],
                    [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    b = np.zeros(4).astype(np.float32)
    scale = np.ones(4).astype(np.float32)

    epsilon = 0.2
    num_channels = 4
    num_groups = 1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        (o_y, o_mean, o_var) = builder.aiGraphcore.groupnormalization(
            [i1, iScale, iB], num_groups, epsilon)
        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_var)
        return [
            o_y, o_mean, o_var,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + iScale,
            poponnx.reservedGradientPrefix() + iB,
            poponnx.reservedGradientPrefix() + o_y
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
        _var = v.var(-1, unbiased=False)

        return [_y, _mean, _var, _input.grad, m.weight.grad, m.bias.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')
