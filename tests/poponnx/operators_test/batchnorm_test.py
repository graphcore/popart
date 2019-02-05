import numpy as np
import pytest
import poponnx
import torch
from op_tester import op_tester

# The calculation for running_mean & running_variance is different
# for onnx and pytorch
# ONNX : running_mean = running_mean * momentum + mean * (1 - momentum)
# PyTorch : running_mean = mean * momentum + running_mean * (1 - momentum)
#
# https://pytorch.org/docs/stable/nn.html?highlight=batchnorm2d#torch.nn.BatchNorm2d
# https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization


def test_batchnorm_train_0_errorcases(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    d2 = np.array([[1, 1], [1, 1]], dtype=np.float32)

    scale = np.ones(2).astype(np.float32)
    scale_2 = np.ones(1).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    b_2 = np.zeros(1).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    mean_2 = np.zeros(1).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    var_2 = np.ones(1).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder_case0(builder):

        i1 = builder.addInputTensor(d2)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)

        return [o_y]

    def init_builder_case1(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale_2)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)

        return [o_y]

    def init_builder_case2(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b_2)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)

        return [o_y]

    def init_builder_case3(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean_2)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)

        return [o_y]

    def init_builder_case4(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var_2)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)

        return [o_y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']

    # Case 0 input tensor has less than 4 dimensions
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case0, None, 'train')

    assert ("batch norm requires" in e_info.value.args[0])

    # Case 1 scale does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case1, None, 'train')

    assert (
        e_info.value.args[0] ==
        "batch norm scale dimension 0 (1) does not equal x dimension 1 (2)")

    # Case 2 b does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case2, None, 'train')

    assert (e_info.value.args[0] ==
            "batch norm b dimension 0 (1) does not equal x dimension 1 (2)")

    # Case 3 mean does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case3, None, 'train')

    assert (e_info.value.args[0] ==
            "batch norm mean dimension 0 (1) does not equal x dimension 1 (2)")

    # Case 4 var does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case4, None, 'train')

    assert (e_info.value.args[0] ==
            "batch norm var dimension 0 (1) does not equal x dimension 1 (2)")


def test_batchnorm_train_0(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    scale = np.ones(2).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o_y, o_mean, o_var, o_smean, o_svar = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)

        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_var)

        return [o_y, 'd__' + i1, 'd__' + o_y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=True)
        _weight = torch.tensor(scale, requires_grad=True)
        _bias = torch.tensor(b, requires_grad=True)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        return [_y, _input.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_batchnorm_train_1(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    # Relax the relative tolerance as small numbers lose precison
    op_tester.rtol = 1e-04

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o_y, o_mean, o_var, o_smean, o_svar = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)

        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_var)
        return [o_y, 'd__' + i1, 'd__' + o_y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        _mean = m.state_dict()['running_mean']
        _var = m.state_dict()['running_var']

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        return [_y, _input.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_batchnorm_train_2(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)

    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    # Relax the relative tolerance as small numbers lose precison
    op_tester.rtol = 1e-04

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o_y, o_mean, o_var, o_smean, o_svar = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)

        for x in range(15):
            o_y, o_mean, o_var, o_smean, o_svar = builder.aiOnnx.batchnormalization(
                [o_y, iScale, iB, o_mean, o_var], 5, epsilon, momentum)

        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_var)
        return [o_y, o_mean, o_var]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm3d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        for x in range(15):
            _y = m(_y)

        _mean = m.state_dict()['running_mean']
        _var = m.state_dict()['running_var']

        return [_y, None, None]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


# This test is a error case where the batch norm in the model is defined as testing but
# the user has performed a train on the model
def test_batchnorm_train_3(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    # Relax the relative tolerance as small numbers lose precison
    op_tester.rtol = 1e-04

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o_y, = builder.aiOnnx.batchnormalization([i1, iScale, iB, iMean, iVar],
                                                 1, epsilon, momentum)

        builder.addOutputTensor(o_y)
        return [o_y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        _mean = m.state_dict()['running_mean']
        _var = m.state_dict()['running_var']

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        return [_y, _input.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert ("Invalid configuration of gradOp" in e_info.value.args[0])


# This test does not work as the inputs are now
# rejects as the mean/var do not match
# input.{C}
# def test_batchnorm_train_3(op_tester):
#     # create test data
#     d1 = np.random.rand(0, 0, 0, 0).astype(np.float32)
#     scale = np.random.rand(0).astype(np.float32)
#     b = np.random.rand(0).astype(np.float32)
#     mean = np.zeros(1).astype(np.float32)
#     var = np.ones(1).astype(np.float32)
#     epsilon = 1e-05
#     momentum = 0.1

#     def init_builder(builder):

#         i1 = builder.addInputTensor(d1)
#         iScale = builder.addInputTensor(scale)
#         iB = builder.addInputTensor(b)
#         iMean = builder.addInputTensor(mean)
#         iVar = builder.addInputTensor(var)
#         o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
#                                                epsilon, momentum)
#         builder.addOutputTensor(o.y)
#         builder.addOutputTensor(o.mean)
#         builder.addOutputTensor(o.var)
#         return [o.y, o.mean, o.var]

#     def reference(ref_data):
#         _input = torch.tensor(d1, requires_grad=False)

#         return [_input, None, None]

#     op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
#     op_tester.run(init_builder, reference, 'train')


def test_batchnorm_test_0(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    scale = np.ones(2).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)
        builder.addOutputTensor(o_y)
        return [o_y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()

        _y = m(_input)

        return [_y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_1(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)
        builder.addOutputTensor(o_y)
        return [o_y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()

        _y = m(_input)

        return [_y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_2(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)
        builder.addOutputTensor(o_y)
        return [o_y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm3d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()
        _y = m(_input)

        return [_y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_3(op_tester):
    # create test data
    d1 = np.random.rand(0, 0, 0, 0).astype(np.float32)
    scale = np.random.rand(0).astype(np.float32)
    b = np.random.rand(0).astype(np.float32)
    mean = np.zeros(0).astype(np.float32)
    var = np.ones(0).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum)
        builder.addOutputTensor(o_y)
        return [o_y]

    def reference(ref_data):
        # In the case the output should match in the input,
        # torch does not like a all zero input
        _input = torch.tensor(d1, requires_grad=False)
        return [_input]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.check_shapes = False
    op_tester.run(init_builder, reference, 'infer')
