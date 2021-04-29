# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import os
from op_tester import op_tester
import platform
import test_util as tu

if platform.system() != "Darwin":
    import onnx

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
    scale_3 = np.ones((2, 2, 2)).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    b_2 = np.zeros(1).astype(np.float32)
    b_3 = np.zeros((2, 2, 2)).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    mean_2 = np.zeros(1).astype(np.float32)
    mean_3 = np.zeros((2, 2, 2)).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    var_2 = np.ones(1).astype(np.float32)
    var_3 = np.ones((2, 2, 4)).astype(np.float32)
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

    def init_builder_case5(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale_3)
        iB = builder.addInputTensor(b_3)
        iMean = builder.addInputTensor(mean_3)
        iVar = builder.addInputTensor(var_3)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum, spatial=0)

        return [o_y]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)

    # Case 0 input tensor has less than 4 dimensions
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_case0, None, 'train')

    assert ("batch norm requires" in e_info.value.args[0])

    # Case 1 scale does not have the size as x.dim(1)
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_case1, None, 'train')

    assert (all([
        msg in e_info.value.args[0]
        for msg in ["expected shape", "scale", "to be [2]"]
    ]))

    # Case 2 b does not have the size as x.dim(1)
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_case2, None, 'train')

    assert (all([
        msg in e_info.value.args[0]
        for msg in ["expected shape", "B", "to be [2]"]
    ]))

    # Case 3 mean does not have the size as x.dim(1)
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_case3, None, 'train')

    assert (all([
        msg in e_info.value.args[0]
        for msg in ["expected shape", "mean", "to be [2]"]
    ]))

    # Case 4 var does not have the size as x.dim(1)
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_case4, None, 'train')

    assert (all([
        msg in e_info.value.args[0]
        for msg in ["expected shape", "var", "to be [2]"]
    ]))

    # Case 5 spacial=False and scale is wrong (note spatial no longer exists in later ONNX versions).
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_case5,
                      None,
                      'train',
                      opsets={
                          "ai.onnx": 7,
                          "ai.onnx.ml": 1,
                          "ai.graphcore": 1
                      })

    assert (all([
        msg in e_info.value.args[0]
        for msg in ["expected shape", "var", "to be [2 2 2]"]
    ]))


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

        return [
            o_y,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=True)
        _weight = torch.tensor(scale, requires_grad=True)
        _bias = torch.tensor(b, requires_grad=True)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        return [_y, _input.grad, d__o]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.atol *= 10
    op_tester.run(init_builder, reference, 'train')


def test_batchnorm_train_1(op_tester):

    # TODO see T7024
    # returning early as this test requires import onnx
    # which causes failure on mac/os.
    # (currently seen on OS/X buildbot)

    if platform.system() == "Darwin":
        print("T7024 : skipping this test on mac/os")
        return
    else:

        # create test data
        d1 = np.random.rand(2, 2, 2, 2).astype(np.float32)
        scale = np.random.rand(2).astype(np.float32)
        b = np.random.rand(2).astype(np.float32)
        mean = np.array([5, 5]).astype(np.float32)
        var = np.array([7, 7]).astype(np.float32)
        epsilon = 1e-05
        momentum = 0.1

        # Relax the relative tolerance as small numbers lose precison
        op_tester.rtol = 1e-04

        initializers = {}

        def init_builder(builder):
            nonlocal initializers

            i1 = builder.addInputTensor(d1)
            iScale = builder.addInputTensor(scale)
            iB = builder.addInputTensor(b)
            iMean = builder.addInitializedInputTensor(mean)
            initializers[iMean] = mean
            iVar = builder.addInitializedInputTensor(var)
            initializers[iVar] = var
            o_y, o_mean, o_var, o_smean, o_svar = builder.aiOnnx.batchnormalization(
                [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)

            builder.addOutputTensor(o_y)
            builder.addOutputTensor(o_mean)
            builder.addOutputTensor(o_var)

            return [
                o_y,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + o_y
            ]

        def reference(ref_data):
            _input = torch.tensor(d1, requires_grad=False)
            _weight = torch.tensor(scale, requires_grad=False)
            _bias = torch.tensor(b, requires_grad=False)
            _mean = torch.tensor(mean, requires_grad=False)
            _var = torch.tensor(var, requires_grad=False)

            m = torch.nn.BatchNorm2d(2,
                                     eps=epsilon,
                                     momentum=momentum,
                                     track_running_stats=True)
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

        op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                              enableRuntimeAsserts=False)
        session = op_tester.run(init_builder, reference, 'train')

        onnx_filename = "test_batchnorm_train_1.onnx"

        session.modelToHost(onnx_filename)
        onnx_model = onnx.load(onnx_filename)

        # Verify that one of the initializers has been updated
        for init in onnx_model.graph.initializer:
            as_numpy = np.array(init.float_data, dtype=np.float32)
            print(f'Checking {init.name} has been updated')
            assert not np.allclose(initializers[init.name], as_numpy)

        os.remove(onnx_filename)


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

        m = torch.nn.BatchNorm3d(2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)
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

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


# This test is a error case where the batch
# norm in the model is defined as testing but
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

        m = torch.nn.BatchNorm2d(2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)
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

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert ("The Ir is in training mode" in e_info.value.args[0])


def test_batchnorm_train_nonspatial(op_tester):
    # Test equivalence of non-spatial (spatial=0) batchnorm by comparing to a pytorch operation (training).
    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    scale = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)
    mean = np.random.rand(2, 2).astype(np.float32)
    var = np.ones((2, 2)).astype(
        np.float32) + (np.random.rand(2, 2).astype(np.float32) - 0.5)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)

        o_y, o_mean, o_var, _, _ = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum, spatial=0)

        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_var)

        return [
            o_y,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o_y
        ]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=True)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        # Convert shapes so we can run BatchNormal1d to mimic spatial=False behaviour.
        _input_bn1dshape = torch.reshape(_input, [2, 2 * 2])
        _weight_bn1dshape = torch.reshape(_weight, [2 * 2])
        _bias_bn1dshape = torch.reshape(_bias, [2 * 2])
        _mean_bn1dshape = torch.reshape(_mean, [2 * 2])
        _var_bn1dshape = torch.reshape(_var, [2 * 2])

        m = torch.nn.BatchNorm1d(2 * 2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)

        m.state_dict()['weight'].copy_(_weight_bn1dshape)
        m.state_dict()['bias'].copy_(_bias_bn1dshape)
        m.state_dict()['running_var'].copy_(_var_bn1dshape)
        m.state_dict()['running_mean'].copy_(_mean_bn1dshape)

        m.train()
        _y_bn1dshape = m(_input_bn1dshape)
        _y = torch.reshape(_y_bn1dshape, [2, 2, 2])

        _mean = m.state_dict()['running_mean']
        _var = m.state_dict()['running_var']

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        _y.backward(d__o)

        return [_y, _input.grad, d__o]

    # TODO: See T21876. My suspicion is that the inaccuracy in calculation is caused by more
    # than lack of transitivity in floating point operations. There seems to be a difference
    # the output of the forward pass when the operation has 5 outputs (it's fine with 1). Set
    # x=random, scale=1, bias=0, mean=0 and variance=1 to see this exaggerated.
    op_tester.atol = 1e-06
    op_tester.rtol = 1e-03
    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder,
                  reference,
                  'train',
                  opsets={
                      "ai.onnx": 7,
                      "ai.onnx.ml": 1,
                      "ai.graphcore": 1
                  })


def test_batchnorm_train_nonspatial_2(op_tester):
    # Test equivalence of non-spatial (spatial=0) batchnorm by comparing to a re-shaped spatial (spatial=1) operation (training).

    # NOTE: op_tester doesn't lend itself to comparing two popart computations with one another. That is,
    # the reference function does not pass in a builder object to allow us to build an alternative graph.
    # Instead of changing op_tester we build do both computations in init_builder, output them both, and
    # in reference we ensure they are compared against one another.

    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    scale = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)
    mean = np.random.rand(2, 2).astype(np.float32)
    var = np.ones((2, 2)).astype(
        np.float32) + (np.random.rand(2, 2).astype(np.float32) - 0.5)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)

        # Batchnorm with spatial=0
        o_y, o_mean, o_var, _, _ = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum, spatial=0)

        builder.addOutputTensor(o_y)
        builder.addOutputTensor(o_mean)
        builder.addOutputTensor(o_var)

        # Batchnorm with spatial=1 but with reshaping such that it should behave the same.
        inout_shape = [2, 2, 2]
        inout_shape_spatial = [2, 4, 1]
        param_shape = [2, 2]
        param_shape_spatial = [4]

        i1_spatial = builder.reshape_const(builder.aiOnnx, [i1],
                                           inout_shape_spatial)
        iScale_spatial = builder.reshape_const(builder.aiOnnx, [iScale],
                                               param_shape_spatial)
        iB_spatial = builder.reshape_const(builder.aiOnnx, [iB],
                                           param_shape_spatial)
        iMean_spatial = builder.reshape_const(builder.aiOnnx, [iMean],
                                              param_shape_spatial)
        iVar_spatial = builder.reshape_const(builder.aiOnnx, [iVar],
                                             param_shape_spatial)
        o_y2_spatial, o_mean2_spatial, o_var2_spatial, _, _ = builder.aiOnnx.batchnormalization(
            [
                i1_spatial, iScale_spatial, iB_spatial, iMean_spatial,
                iVar_spatial
            ],
            5,
            epsilon,
            momentum,
            spatial=1)
        o_y2 = builder.reshape_const(builder.aiOnnx, [o_y2_spatial],
                                     inout_shape)
        o_mean2 = builder.reshape_const(builder.aiOnnx, [o_mean2_spatial],
                                        param_shape)
        o_var2 = builder.reshape_const(builder.aiOnnx, [o_var2_spatial],
                                       param_shape)
        builder.addOutputTensor(o_y2)
        builder.addOutputTensor(o_mean2)
        builder.addOutputTensor(o_var2)

        return [o_y, o_mean, o_var, o_y2, o_mean2, o_var2]

    def reference(ref_data):
        return [
            ref_data.getOutputTensor(3),
            ref_data.getOutputTensor(4),
            ref_data.getOutputTensor(5),
            ref_data.getOutputTensor(0),
            ref_data.getOutputTensor(1),
            ref_data.getOutputTensor(2)
        ]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 7,
                      "ai.onnx.ml": 1,
                      "ai.graphcore": 1
                  })


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

#     op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'], enableRuntimeAsserts=False)
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

        m = torch.nn.BatchNorm2d(2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()

        _y = m(_input)

        return [_y]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_1(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.random.rand(2).astype(np.float32)
    var = np.ones(2).astype(
        np.float32) + (np.random.rand(2).astype(np.float32) - 0.5)
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

        m = torch.nn.BatchNorm2d(2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()

        _y = m(_input)

        return [_y]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_2(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.random.rand(2).astype(np.float32)
    var = np.ones(2).astype(
        np.float32) + (np.random.rand(2).astype(np.float32) - 0.5)
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

        m = torch.nn.BatchNorm3d(2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()
        _y = m(_input)

        return [_y]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
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

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.check_shapes = False
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_nonspatial(op_tester):
    # Test equivalence of non-spatial (spatial=0) batchnorm by comparing to a pytorch operation (inference).
    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    scale = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)
    mean = np.random.rand(2, 2).astype(np.float32)
    var = np.ones((2, 2)).astype(
        np.float32) + (np.random.rand(2, 2).astype(np.float32) - 0.5)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        (o_y, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum, spatial=0)
        builder.addOutputTensor(o_y)
        return [o_y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        _input = torch.reshape(_input, [2, 2 * 2])
        _weight = torch.reshape(_weight, [2 * 2])
        _bias = torch.reshape(_bias, [2 * 2])
        _mean = torch.reshape(_mean, [2 * 2])
        _var = torch.reshape(_var, [2 * 2])

        m = torch.nn.BatchNorm1d(2 * 2,
                                 eps=epsilon,
                                 momentum=momentum,
                                 track_running_stats=True)

        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_var'].copy_(_var)
        m.state_dict()['running_mean'].copy_(_mean)

        m.eval()  # turn off training.
        _y = m(_input)
        _y = torch.reshape(_y, [2, 2, 2])

        return [_y]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 7,
                      "ai.onnx.ml": 1,
                      "ai.graphcore": 1
                  })


def test_batchnorm_test_nonspatial_2(op_tester):
    # Test equivalence of non-spatial (spatial=0) batchnorm by comparing to a re-shaped spatial (spatial=1) operation (inference).

    # NOTE: op_tester doesn't lend itself to comparing two popart computations with one another. That is,
    # the reference function does not pass in a builder object to allow us to build an alternative graph.
    # Instead of changing op_tester we build do both computations in init_builder, output them both, and
    # in reference we ensure they are compared against one another.

    d1 = np.random.rand(2, 2, 2).astype(np.float32)
    scale = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)
    mean = np.random.rand(2, 2).astype(np.float32)
    var = np.ones((2, 2)).astype(
        np.float32) + (np.random.rand(2, 2).astype(np.float32) - 0.5)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)

        # Batchnorm with spatial=0
        (o_y1, ) = builder.aiOnnx.batchnormalization(
            [i1, iScale, iB, iMean, iVar], 1, epsilon, momentum, spatial=0)
        builder.addOutputTensor(o_y1)

        # Batchnorm with spatial=1 but with reshaping such that it should behave the same.
        inout_shape = [2, 2, 2]
        inout_shape_spatial = [2, 4, 1]
        param_shape_spatial = [4]

        i1_spatial = builder.reshape_const(builder.aiOnnx, [i1],
                                           inout_shape_spatial)
        iScale_spatial = builder.reshape_const(builder.aiOnnx, [iScale],
                                               param_shape_spatial)
        iB_spatial = builder.reshape_const(builder.aiOnnx, [iB],
                                           param_shape_spatial)
        iMean_spatial = builder.reshape_const(builder.aiOnnx, [iMean],
                                              param_shape_spatial)
        iVar_spatial = builder.reshape_const(builder.aiOnnx, [iVar],
                                             param_shape_spatial)
        (o_y2_spatial, ) = builder.aiOnnx.batchnormalization([
            i1_spatial, iScale_spatial, iB_spatial, iMean_spatial, iVar_spatial
        ],
                                                             1,
                                                             epsilon,
                                                             momentum,
                                                             spatial=1)
        o_y2 = builder.reshape_const(builder.aiOnnx, [o_y2_spatial],
                                     inout_shape)
        builder.addOutputTensor(o_y2)
        return [o_y1, o_y2]

    def reference(ref_data):
        return [ref_data.getOutputTensor(1), ref_data.getOutputTensor(0)]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 7,
                      "ai.onnx.ml": 1,
                      "ai.graphcore": 1
                  })


# Run the inference model multiple times and test that the outputs
# are the same.
def test_batchnorm_repeated():
    # create test data
    d1 = np.random.rand(1, 3, 2, 2).astype(np.float16) * 100
    scale = np.random.rand(3).astype(np.float16)
    b = np.random.rand(3).astype(np.float16)
    mean = np.random.rand(3).astype(np.float16)
    var = np.random.rand(3).astype(np.float16)
    epsilon = 1e-05
    momentum = 0.1

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1))
    iScale = builder.addInitializedInputTensor(scale)
    iB = builder.addInitializedInputTensor(b)
    iMean = builder.addInitializedInputTensor(mean)
    iVar = builder.addInitializedInputTensor(var)
    (o_y, ) = builder.aiOnnx.batchnormalization([i1, iScale, iB, iMean, iVar],
                                                1, epsilon, momentum)
    builder.addOutputTensor(o_y)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o_y: popart.AnchorReturnType("All")})

    device = tu.create_test_device()

    options = popart.SessionOptions()
    options.enableStochasticRounding = False

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=device,
                                      userOptions=options)

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {i1: d1}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)
    first_result = np.copy(anchors[o_y])

    for i in range(0, 10):
        stepio = popart.PyStepIO(inputs, anchors)
        session.run(stepio)

        assert np.allclose(first_result, np.copy(anchors[o_y])) == True


def test_batchnorm_train_half_fp32var(op_tester):
    # create test data
    d1 = np.random.rand(1, 3, 2, 2).astype(np.float16) * 100
    scale = np.random.rand(3).astype(np.float16)
    b = np.random.rand(3).astype(np.float16)
    mean = np.random.rand(3).astype(np.float16)
    var = np.random.rand(3).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1))
    iScale = builder.addInitializedInputTensor(scale)
    iB = builder.addInitializedInputTensor(b)
    iMean = builder.addInitializedInputTensor(mean)
    iVar = builder.addInitializedInputTensor(var)
    o_y, o_mean, o_var, o_smean, o_svar = builder.aiOnnx.batchnormalization(
        [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)
    builder.addOutputTensor(o_y)
    lossId = builder.aiGraphcore.identityloss([o_y])
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o_y: popart.AnchorReturnType("All")})

    device = tu.create_test_device()

    options = popart.SessionOptions()
    options.enableStochasticRounding = False

    session = popart.TrainingSession(fnModel=proto,
                                     loss=lossId,
                                     dataFlow=dataFlow,
                                     deviceInfo=device,
                                     optimizer=popart.ConstSGD(0.01),
                                     userOptions=options)

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {i1: d1}
    stepio = popart.PyStepIO(inputs, anchors)
    session.weightsFromHost()
    session.run(stepio)
    stepio = popart.PyStepIO(inputs, anchors)
    session.run(stepio)


def test_batchnorm_inference_half_fp32var(op_tester):
    # create test data
    d1 = np.random.rand(1, 3, 2, 2).astype(np.float16) * 100
    scale = np.random.rand(3).astype(np.float16)
    b = np.random.rand(3).astype(np.float16)
    mean = np.random.rand(3).astype(np.float16)
    var = np.random.rand(3).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1))
    iScale = builder.addInitializedInputTensor(scale)
    iB = builder.addInitializedInputTensor(b)
    iMean = builder.addInitializedInputTensor(mean)
    iVar = builder.addInitializedInputTensor(var)
    (o_y, ) = builder.aiOnnx.batchnormalization([i1, iScale, iB, iMean, iVar],
                                                1, epsilon, momentum)
    builder.addOutputTensor(o_y)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o_y: popart.AnchorReturnType("All")})

    device = tu.create_test_device()

    options = popart.SessionOptions()
    options.enableStochasticRounding = False

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=device,
                                      userOptions=options)

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {i1: d1}
    stepio = popart.PyStepIO(inputs, anchors)
    session.run(stepio)
    stepio = popart.PyStepIO(inputs, anchors)
    session.run(stepio)


def test_batchnorm_shapeinference(op_tester):
    # create test data
    d1 = np.random.rand(1, 3, 2, 2).astype(np.float32) * 100
    scale = np.random.rand(3).astype(np.float32)
    b = np.random.rand(3).astype(np.float32)
    mean = np.random.rand(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1
    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1))
    iScale = builder.addInitializedInputTensor(scale)
    iB = builder.addInitializedInputTensor(b)
    iMean = builder.addInitializedInputTensor(mean)
    iVar = builder.addInitializedInputTensor(var)
    o_y, o_mean, o_var, o_smean, o_svar = builder.aiOnnx.batchnormalization(
        [i1, iScale, iB, iMean, iVar], 5, epsilon, momentum)
    builder.addOutputTensor(o_y)
    builder.addOutputTensor(o_mean)
    builder.addOutputTensor(o_var)
    builder.addOutputTensor(o_smean)
    builder.addOutputTensor(o_svar)
    lossId = builder.aiGraphcore.identityloss([o_y])
    proto = builder.getModelProto()
    anchors = [o_y, o_mean, o_var, o_smean, o_svar]
    art = popart.AnchorReturnType("All")
    dataFlow = popart.DataFlow(1, {a: art for a in anchors})
    device = tu.create_test_device()
    options = popart.SessionOptions()
    options.enableStochasticRounding = False
    # store the shapes here to make sure we are checking shapes
    #  before the IR is complete (i.e. testing onnx shape inference)
    shapes = []
    for a in anchors:
        shapes.append(tuple(builder.getTensorShape(a)))
    session = popart.TrainingSession(fnModel=proto,
                                     loss=lossId,
                                     dataFlow=dataFlow,
                                     deviceInfo=device,
                                     optimizer=popart.ConstSGD(0.01),
                                     userOptions=options)
    anchors = session.initAnchorArrays()
    session.prepareDevice()
    inputs = {i1: d1}
    stepio = popart.PyStepIO(inputs, anchors)
    session.weightsFromHost()
    session.run(stepio)
    stepio = popart.PyStepIO(inputs, anchors)
    # This tests the shape inference has run
    for a, b in zip([o_y, o_mean, o_var, o_smean, o_svar], shapes):
        assert anchors[a].shape == b
