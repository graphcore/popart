# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import pytest
import popart
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.random as npr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

import torch_lamb


def compare_against_pytorch(optType, optMaps, batchesPerStep=5, scaled=False):
    seed = 1015
    npr.seed(seed)
    torch.manual_seed(seed)

    optkwargs = {}

    if optType == "adam":
        popartOpt = popart.Adam
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.L2Regularization
        optkwargs["scaled_optimizer_state"] = scaled
    elif optType == "adamw":
        popartOpt = popart.Adam
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.Decay
        optkwargs["scaled_optimizer_state"] = scaled
    elif optType == "adamax":
        popartOpt = popart.Adam
        optkwargs["mode"] = popart.AdamMode.AdaMax
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.L2Regularization
        optkwargs["scaled_optimizer_state"] = scaled
    elif optType == "lamb":
        popartOpt = popart.Adam
        optkwargs["mode"] = popart.AdamMode.Lamb
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.Decay
        optkwargs["scaled_optimizer_state"] = scaled
    elif optType == "lambnobias":
        popartOpt = popart.Adam
        optkwargs["mode"] = popart.AdamMode.LambNoBias
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.Decay
        optkwargs["scaled_optimizer_state"] = scaled
    elif optType == "adagrad":
        popartOpt = popart.Adaptive
        optkwargs["mode"] = popart.AdaptiveMode.AdaGrad
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.L2Regularization
    elif optType == "rmsprop":
        popartOpt = popart.Adaptive
        optkwargs["mode"] = popart.AdaptiveMode.RMSProp
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.L2Regularization
    elif optType == "centeredrmsprop":
        popartOpt = popart.Adaptive
        optkwargs["mode"] = popart.AdaptiveMode.CenteredRMSProp
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.L2Regularization
    elif optType == "adadelta":
        popartOpt = popart.Adaptive
        optkwargs["mode"] = popart.AdaptiveMode.AdaDelta
        optkwargs["weight_decay_mode"] = popart.WeightDecayMode.L2Regularization
    elif optType == "sgd0":
        popartOpt = popart.SGD
    elif optType == "sgd1":
        popartOpt = popart.SGD
        optkwargs[
            "accumulatorAndMomentum"] = popart.SGDAccumulatorAndMomentum.Combined
    elif optType == "sgd2":
        popartOpt = popart.SGD
        optkwargs[
            "accumulatorAndMomentum"] = popart.SGDAccumulatorAndMomentum.Separate
    else:
        raise "Unknown optType: " + optType

    #L1 loss value
    lambda1 = 1.0

    # tensor dimensions and replications
    height = 2
    numberOfSteps = len(optMaps)
    sampleShape = [height, height]
    replicationFactor = 1
    accumulationFactor = 1
    nVirtualGraphs = 1
    samplesPerBatch = 4
    divvyFactor = replicationFactor * accumulationFactor
    samplesPerMicroBatch = samplesPerBatch // divvyFactor
    nIPUs = replicationFactor * nVirtualGraphs
    stepDataShape = [batchesPerStep, samplesPerBatch, height, height]
    microBatchShape = [samplesPerMicroBatch, height, height]
    stepDataInfo = popart.TensorInfo("FLOAT", stepDataShape)
    microBatchInfo = popart.TensorInfo("FLOAT", microBatchShape)

    #initial weight and input values
    w0vals = np.array(npr.randn(height, height), dtype=np.float32)
    w1vals = np.array(npr.randn(height, height), dtype=np.float32)
    inputVals = [
        np.array(npr.randn(*stepDataShape), dtype=np.float32)
        for i in range(numberOfSteps)
    ]

    # Build the ONNX Model
    builder = popart.Builder()
    input0 = builder.addInputTensor(microBatchInfo)
    w0 = builder.addInitializedInputTensor(w0vals)
    w1 = builder.addInitializedInputTensor(w1vals)

    # Model:
    #
    # input  w0     w1
    #     \  |      |
    #      mul  -  add - L1 loss

    mm0 = builder.aiOnnx.mul([input0, w0])
    mm1 = builder.aiOnnx.add([mm0, w1])
    l1 = builder.aiGraphcore.l1loss([mm1], lambda1)
    art = popart.AnchorReturnType("All")
    dataFlow = popart.DataFlow(batchesPerStep, {})
    device = tu.create_test_device(numIpus=nIPUs)
    userOptions = popart.SessionOptions()
    userOptions.enableGradientAccumulation = False
    userOptions.enablePrefetchDatastreams = False

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFlow=dataFlow,
        userOptions=userOptions,
        loss=l1,
        optimizer=popartOpt(optMaps[0], **optkwargs),
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchorArrays = session.initAnchorArrays()

    session.prepareDevice()
    session.weightsFromHost()

    for step in range(numberOfSteps):
        stepio = popart.PyStepIO({input0: inputVals[step]}, anchorArrays)
        session.run(stepio)

        if (step < numberOfSteps - 1):
            session.updateOptimizerFromHost(
                popartOpt(optMaps[step + 1], **optkwargs))

    session.weightsToHost()
    w0R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
    w1R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
    weightsRead = popart.PyWeightsIO({w0: w0R, w1: w1R})
    session.readWeights(weightsRead)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.w0 = torch.nn.Parameter(torch.from_numpy(w0vals.copy()))
            self.w1 = torch.nn.Parameter(torch.from_numpy(w1vals.copy()))

        def forward(self, x, _):  # i is an unused parameter needed in torch
            mm0 = torch.mul(x, self.w0)
            mm1 = torch.add(mm0, self.w1)
            return mm1

    net = Net()

    for step in range(numberOfSteps):
        if step is 0:
            oldOptimizer = None
        else:
            oldOptimizer = optimizer

        if optType == "adam":
            optimizer = optim.Adam(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                betas=(optMaps[step]["defaultBeta1"][0],
                       optMaps[step]["defaultBeta2"][0]),
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0])
        elif optType == "adamw":
            optimizer = optim.AdamW(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                betas=(optMaps[step]["defaultBeta1"][0],
                       optMaps[step]["defaultBeta2"][0]),
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0])
        elif optType == "adamax":
            optimizer = optim.Adamax(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                betas=(optMaps[step]["defaultBeta1"][0],
                       optMaps[step]["defaultBeta2"][0]),
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0])
        elif optType == "lamb":
            optimizer = torch_lamb.Lamb(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                betas=(optMaps[step]["defaultBeta1"][0],
                       optMaps[step]["defaultBeta2"][0]),
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0])
        elif optType == "lambnobias":
            optimizer = torch_lamb.Lamb(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                betas=(optMaps[step]["defaultBeta1"][0],
                       optMaps[step]["defaultBeta2"][0]),
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0],
                biasCorrection=False)
        elif optType == "adagrad":
            optimizer = optim.Adagrad(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0],
                eps=optMaps[step]["defaultEps"][0])
        elif optType == "rmsprop":
            optimizer = optim.RMSprop(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                alpha=optMaps[step]["defaultAlpha"][0],
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0],
                momentum=optMaps[step]["defaultMomentum"][0]
                if "defaultMomentum" in optMaps[step] else 0.0)
        elif optType == "centeredrmsprop":
            optimizer = optim.RMSprop(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                alpha=optMaps[step]["defaultAlpha"][0],
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0],
                momentum=optMaps[step]["defaultMomentum"][0]
                if "defaultMomentum" in optMaps[step] else 0.0,
                centered=True)
        elif optType == "adadelta":
            optimizer = optim.Adadelta(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                rho=optMaps[step]["defaultAlpha"][0],
                eps=optMaps[step]["defaultEps"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0])
        else:  # Same for SGD1 and SGD2.
            optimizer = optim.SGD(
                net.parameters(),
                lr=optMaps[step]["defaultLearningRate"][0],
                momentum=optMaps[step]["defaultMomentum"][0],
                dampening=optMaps[step]["defaultDampening"][0],
                weight_decay=optMaps[step]["defaultWeightDecay"][0])

        if step is 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    param_state = optimizer.state[p][
                        'momentum_buffer'] = p.data * 0
                    param_state = optimizer.state[p]['exp_avg'] = p.data * 0
                    param_state = optimizer.state[p]['exp_avg_sq'] = p.data * 0
                    param_state = optimizer.state[p]['exp_inf'] = p.data * 0
                    param_state = optimizer.state[p]['square_avg'] = p.data * 0
                    param_state = optimizer.state[p]['grad_avg'] = p.data * 0
                    param_state = optimizer.state[p]['acc_delta'] = p.data * 0
                    param_state = optimizer.state[p]['sum'] = p.data * 0
                    param_state = optimizer.state[p]['step'] = 0
        else:
            for group, oldGroup in zip(optimizer.param_groups,
                                       oldOptimizer.param_groups):
                for p, oldp in zip(group['params'], oldGroup['params']):
                    param_state = optimizer.state[p][
                        'momentum_buffer'] = oldOptimizer.state[p][
                            'momentum_buffer']
                    param_state = optimizer.state[p][
                        'exp_avg'] = oldOptimizer.state[p]['exp_avg']
                    param_state = optimizer.state[p][
                        'exp_avg_sq'] = oldOptimizer.state[p]['exp_avg_sq']
                    param_state = optimizer.state[p][
                        'exp_inf'] = oldOptimizer.state[p]['exp_inf']
                    param_state = optimizer.state[p][
                        'square_avg'] = oldOptimizer.state[p]['square_avg']
                    param_state = optimizer.state[p][
                        'grad_avg'] = oldOptimizer.state[p]['grad_avg']
                    param_state = optimizer.state[p][
                        'acc_delta'] = oldOptimizer.state[p]['acc_delta']
                    param_state = optimizer.state[p][
                        'sum'] = oldOptimizer.state[p]['sum']
                    param_state = optimizer.state[p][
                        'step'] = oldOptimizer.state[p]['step']

        for i in range(batchesPerStep):
            out = net(torch.from_numpy(inputVals[step][i]), i)
            loss = lambda1 * torch.mean(torch.abs(out))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    delta0 = np.sum(np.abs(net.w0.detach().numpy() - w0vals))
    delta1 = np.sum(np.abs(net.w1.detach().numpy() - w1vals))
    print("pytorch baseline")
    print("Total moved by w0: ", delta0)
    print("Total moved by w1: ", delta1)

    error0 = np.sum(np.abs(w0R - net.w0.detach().numpy())) / delta0
    error1 = np.sum(np.abs(w1R - net.w1.detach().numpy())) / delta1
    print("without pipelining")
    print("Total moved by w0: ", np.sum(np.abs(w0R - w0vals)))
    print("Total moved by w1: ", np.sum(np.abs(w1R - w1vals)))
    print("l1 error for w0: ", error0)
    print("l1 error for w1: ", error1)
    assert (error0 < 1e-5)
    assert (error1 < 1e-5)


def sgd_test_against_pytorch(optType):
    #optimizer parameters
    defaultLearningRate0 = 0.5
    defaultLearningRate1 = 0.3
    defaultLearningRate2 = 0.1

    defaultMomentum0 = 0
    defaultDampening0 = 0
    if optType != "sgd0":
        defaultMomentum0 = 0.1
        defaultDampening0 = 0.3
    lossScaling0 = 10.0
    defaultVelocityScaling0 = 0.5
    defaultWeightDecay0 = 0.01

    optMap0 = {
        "defaultLearningRate": (defaultLearningRate0, False),
        "defaultMomentum": (defaultMomentum0, defaultMomentum0 == 0),
        "defaultDampening": (defaultDampening0, False),
        "defaultVelocityScaling": (defaultVelocityScaling0, False),
        "lossScaling": (lossScaling0, False),
        "defaultWeightDecay": (defaultWeightDecay0, False)
    }

    optMap1 = {
        "defaultLearningRate": (defaultLearningRate1, False),
        "defaultMomentum": (defaultMomentum0, defaultMomentum0 == 0),
        "defaultDampening": (defaultDampening0, False),
        "defaultVelocityScaling": (defaultVelocityScaling0, False),
        "lossScaling": (lossScaling0, False),
        "defaultWeightDecay": (defaultWeightDecay0, False)
    }

    optMap2 = {
        "defaultLearningRate": (defaultLearningRate2, False),
        "defaultMomentum": (defaultMomentum0, defaultMomentum0 == 0),
        "defaultDampening": (defaultDampening0, False),
        "defaultVelocityScaling": (defaultVelocityScaling0, False),
        "lossScaling": (lossScaling0, False),
        "defaultWeightDecay": (defaultWeightDecay0, False)
    }

    compare_against_pytorch(optType, [optMap0, optMap1, optMap2])


@tu.requires_ipu_model
def test_sgd0_against_pytorch():
    sgd_test_against_pytorch("sgd0")


@tu.requires_ipu_model
def test_sgd2_against_pytorch():
    sgd_test_against_pytorch("sgd2")


@tu.requires_ipu_model
def test_sgd1_against_pytorch():
    """
    Comparison of popart and PyTorch optimizers, and the changes needed to PyTorch
    to match popart. Note that these differences should have no effect on overall
    training measures, and this discussion is just for those interested in exact
    reproducility between popart and pytorch

    The main differences are:
    1)
    pytorch optimizer, in the very first iteration with a new optimizer:
    initializes velocity tensor with zeros and does not do damping,
    popart optimizer, in the first iteration with a new optimizer:
    retains velocity tensor from previous state with previous optimizer,
    or sets it to zero if the first round of training

    2)
    popart and pytorch updates the optimizer at a different "phase"
    a)    v <- v * mm  + (1 - dp) * wd * w
    b)    v <- v + (1 - dp) * g
    c)    w <- w - lr * v

    pytorch goes (abc)(abc)(abc)(abc)
    popart goes  (abca)(bca)(bca)(bca)

    where changes to the optimizer can be done between periods. For this reason,
    updates to mm, dp, and wd have different effects.

    See also sgd_mixed_mode_test_cpp_1_3.cpp
    """

    sgd_test_against_pytorch("sgd1")


@tu.requires_ipu_model
@pytest.mark.parametrize('scaled', (True, False))
def test_adam_against_pytorch(scaled):
    """
    Comparison of popart and PyTorch optimizers (Adam, AdamW, AdaMax, Lamb)
    """

    #optimizer parameters
    defaultLearningRate0 = 0.005
    defaultLearningRate1 = 0.003
    defaultLearningRate2 = 0.001

    defaultBeta1 = 0.7
    defaultBeta2 = 0.8
    defaultWeightDecay = 0.1
    defaultEps = 1e-6
    lossScaling = 10.0

    optMap0 = {
        "defaultLearningRate": (defaultLearningRate0, False),
        "defaultBeta1": (defaultBeta1, False),
        "defaultBeta2": (defaultBeta2, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultEps": (defaultEps, False),
        "lossScaling": (lossScaling, False),
    }

    optMap1 = {
        "defaultLearningRate": (defaultLearningRate1, False),
        "defaultBeta1": (defaultBeta1, False),
        "defaultBeta2": (defaultBeta2, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultEps": (defaultEps, False),
        "lossScaling": (lossScaling, False),
    }

    optMap2 = {
        "defaultLearningRate": (defaultLearningRate2, False),
        "defaultBeta1": (defaultBeta1, False),
        "defaultBeta2": (defaultBeta2, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultEps": (defaultEps, False),
        "lossScaling": (lossScaling, False),
    }

    # Test Adam/AdamW against pytorch
    compare_against_pytorch("adam", [optMap0, optMap1, optMap2], scaled=scaled)
    compare_against_pytorch("adamw", [optMap0, optMap1, optMap2],
                            scaled=scaled)
    # Test Lamb against pytorch (implemented in torch_lamb.py)
    compare_against_pytorch("lamb", [optMap0, optMap1, optMap2], scaled=scaled)
    # Test Lamb without bias correction (V3 paper)
    compare_against_pytorch("lambnobias", [optMap0, optMap1, optMap2],
                            scaled=scaled)
    # Test AdaMax
    compare_against_pytorch("adamax", [optMap0, optMap1, optMap2],
                            scaled=scaled)


@tu.requires_ipu_model
def test_adaptive_against_pytorch():
    """
    Comparison of popart and PyTorch optimizers (AdaGrad, RMSProp, AdaDelta)
    """

    #optimizer parameters
    defaultLearningRate0 = 0.005
    defaultLearningRate1 = 0.003
    defaultLearningRate2 = 0.001

    defaultAlpha = 0.7
    defaultMomentum = 0.8
    defaultWeightDecay = 0.1
    defaultEps = 1e-6
    lossScaling = 10.0

    optMap0 = {
        "defaultLearningRate": (defaultLearningRate0, False),
        "defaultAlpha": (defaultAlpha, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultEps": (defaultEps, False),
        "lossScaling": (lossScaling, False),
    }

    optMap1 = {
        "defaultLearningRate": (defaultLearningRate1, False),
        "defaultAlpha": (defaultAlpha, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultEps": (defaultEps, False),
        "lossScaling": (lossScaling, False),
    }

    optMap2 = {
        "defaultLearningRate": (defaultLearningRate2, False),
        "defaultAlpha": (defaultAlpha, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultEps": (defaultEps, False),
        "lossScaling": (lossScaling, False),
    }

    # Test variants with momentum
    optMap0m = optMap0.copy()
    optMap0m["defaultMomentum"] = (defaultMomentum, False)
    optMap1m = optMap1.copy()
    optMap1m["defaultMomentum"] = (defaultMomentum, False)
    optMap2m = optMap2.copy()
    optMap2m["defaultMomentum"] = (defaultMomentum, False)

    compare_against_pytorch("adagrad", [optMap0, optMap1, optMap2])
    compare_against_pytorch("rmsprop", [optMap0, optMap1, optMap2])
    compare_against_pytorch("rmsprop", [optMap0m, optMap1m, optMap2m])
    compare_against_pytorch("centeredrmsprop", [optMap0, optMap1, optMap2])
    compare_against_pytorch("centeredrmsprop", [optMap0m, optMap1m, optMap2m])
    compare_against_pytorch("adadelta", [optMap0, optMap1, optMap2])
